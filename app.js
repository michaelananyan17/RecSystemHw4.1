class MovieLensApp {
    constructor() {
        // Data stores
        this.interactions = [];
        this.items = new Map(); // itemId -> {title, year, genres}
        this.users = new Map(); // userId -> {age, gender, occupation}
        this.occupationVocab = new Map();

        // Mappings
        this.userMap = new Map();
        this.itemMap = new Map();
        this.reverseUserMap = new Map();
        this.reverseItemMap = new Map();

        // Pre-computed user data
        this.userTopRated = new Map();
        this.qualifiedUsers = [];
        
        // Models
        this.simpleModel = null;
        this.deepModel = null;
        
        // Config
        this.config = {
            maxInteractions: 80000,
            embeddingDim: 32,
            batchSize: 512,
            epochs: 15, // Reduced slightly for faster demo training
            learningRate: 0.001
        };
        
        // State
        this.lossHistory = [];
        this.isTraining = false;
        
        this.initializeUI();
    }
    
    initializeUI() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('train').addEventListener('click', () => this.train());
        document.getElementById('test').addEventListener('click', () => this.test());
        this.updateStatus('Click "Load Data" to start');
    }
    
    async loadData() {
        this.updateStatus('Loading data... (interactions, items, users)');
        
        try {
            // Fetch all data in parallel
            const [interactionsResponse, itemsResponse, usersResponse] = await Promise.all([
                fetch('data/u.data'),
                fetch('data/u.item'),
                fetch('data/u.user')
            ]);

            const interactionsText = await interactionsResponse.text();
            const itemsText = await itemsResponse.text();
            const usersText = await usersResponse.text();

            // Parse interactions
            this.interactions = interactionsText.trim().split('\n').slice(0, this.config.maxInteractions).map(line => {
                const [userId, itemId, rating, timestamp] = line.split('\t');
                return { userId: parseInt(userId), itemId: parseInt(itemId), rating: parseFloat(rating), timestamp: parseInt(timestamp) };
            });

            // Parse items and genres
            const genres = ['unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'];
            itemsText.trim().split('\n').forEach(line => {
                const parts = line.split('|');
                const itemId = parseInt(parts[0]);
                const title = parts[1];
                const yearMatch = title.match(/\((\d{4})\)$/);
                const year = yearMatch ? parseInt(yearMatch[1]) : null;
                const itemGenres = parts.slice(5).map(g => parseInt(g));

                this.items.set(itemId, {
                    title: title.replace(/\(\d{4}\)$/, '').trim(),
                    year: year,
                    genres: itemGenres
                });
            });

            // Parse users and build occupation vocabulary
            const occupations = new Set();
            usersText.trim().split('\n').forEach(line => {
                const [userId, age, gender, occupation] = line.split('|');
                this.users.set(parseInt(userId), {
                    age: parseInt(age) / 100, // Simple normalization
                    gender: gender === 'M' ? 1 : 0,
                    occupation: occupation
                });
                occupations.add(occupation);
            });
            Array.from(occupations).sort().forEach((occ, i) => this.occupationVocab.set(occ, i));

            // Create mappings and find qualified users
            this.createMappings();
            this.findQualifiedUsers();
            
            this.updateStatus(`Loaded ${this.interactions.length} interactions, ${this.items.size} items, ${this.users.size} users. Ready to train.`);
            document.getElementById('train').disabled = false;
            
        } catch (error) {
            this.updateStatus(`Error loading data: ${error.message}. Make sure u.data, u.item, and u.user are in a 'data' folder.`);
        }
    }
    
    createMappings() {
        const userSet = new Set(this.interactions.map(i => i.userId));
        const itemSet = new Set(this.interactions.map(i => i.itemId));
        
        Array.from(userSet).forEach((userId, index) => {
            this.userMap.set(userId, index);
            this.reverseUserMap.set(index, userId);
        });
        
        Array.from(itemSet).forEach((itemId, index) => {
            this.itemMap.set(itemId, index);
            this.reverseItemMap.set(index, itemId);
        });
        
        const userInteractions = new Map();
        this.interactions.forEach(interaction => {
            const userId = interaction.userId;
            if (!userInteractions.has(userId)) userInteractions.set(userId, []);
            userInteractions.get(userId).push(interaction);
        });
        
        userInteractions.forEach((interactions, userId) => {
            interactions.sort((a, b) => b.rating - a.rating || b.timestamp - a.timestamp);
        });
        this.userTopRated = userInteractions;
    }
    
    findQualifiedUsers() {
        this.qualifiedUsers = [];
        this.userTopRated.forEach((interactions, userId) => {
            if (interactions.length >= 20) {
                this.qualifiedUsers.push(userId);
            }
        });
    }

    getUserFeatures(userId) {
        const user = this.users.get(userId);
        const occupationIndex = this.occupationVocab.get(user.occupation);
        const occupationOneHot = Array(this.occupationVocab.size).fill(0);
        if (occupationIndex !== undefined) occupationOneHot[occupationIndex] = 1;
        return [user.age, user.gender, ...occupationOneHot];
    }
    
    async train() {
        if (this.isTraining) return;
        
        this.isTraining = true;
        document.getElementById('train').disabled = true;
        this.lossHistory = [];
        this.updateStatus('Initializing models...');

        // Initialize Simple Model
        this.simpleModel = new SimpleTwoTowerModel(this.userMap.size, this.itemMap.size, this.config.embeddingDim);
        this.simpleModel.optimizer.learningRate = this.config.learningRate;
        
        // Initialize Deep Model
        const userFeatureDim = this.getUserFeatures(this.interactions[0].userId).length;
        const itemFeatureDim = this.items.get(this.interactions[0].itemId).genres.length;

        this.deepModel = new DeepTwoTowerModel(
            this.userMap.size, this.itemMap.size, this.config.embeddingDim,
            userFeatureDim, itemFeatureDim, [64, this.config.embeddingDim]
        );
        this.deepModel.optimizer.learningRate = this.config.learningRate;
        
        // Prepare training data indices
        const userIndices = this.interactions.map(i => this.userMap.get(i.userId));
        const itemIndices = this.interactions.map(i => this.itemMap.get(i.itemId));
        
        this.updateStatus('Starting training for both models...');
        const numBatches = Math.ceil(userIndices.length / this.config.batchSize);
        
        for (let epoch = 0; epoch < this.config.epochs; epoch++) {
            let epochLossSimple = 0;
            let epochLossDeep = 0;
            
            for (let batch = 0; batch < numBatches; batch++) {
                const start = batch * this.config.batchSize;
                const end = Math.min(start + this.config.batchSize, userIndices.length);
                
                // --- Simple Model Training Step ---
                const batchUsersSimple = userIndices.slice(start, end);
                const batchItemsSimple = itemIndices.slice(start, end);
                const lossSimple = await this.simpleModel.trainStep(batchUsersSimple, batchItemsSimple);
                epochLossSimple += lossSimple;

                // --- Deep Model Training Step ---
                const batchUserIds = batchUsersSimple.map(u_idx => this.reverseUserMap.get(u_idx));
                const batchItemIds = batchItemsSimple.map(i_idx => this.reverseItemMap.get(i_idx));

                const userFeatures = batchUserIds.map(uid => this.getUserFeatures(uid));
                const itemFeatures = batchItemIds.map(iid => this.items.get(iid).genres);

                const lossDeep = await this.deepModel.trainStep(batchUsersSimple, userFeatures, batchItemsSimple, itemFeatures);
                epochLossDeep += lossDeep;
                
                this.lossHistory.push(lossDeep); // Plot deep model's loss
                this.updateLossChart();
                
                if (batch % 10 === 0) {
                    this.updateStatus(`Epoch ${epoch + 1}/${this.config.epochs}, Batch ${batch}/${numBatches}, Deep Loss: ${lossDeep.toFixed(4)}`);
                }
                
                await tf.nextFrame(); // Allow UI to update
            }
            
            this.updateStatus(`Epoch ${epoch + 1} done. Avg Simple Loss: ${(epochLossSimple/numBatches).toFixed(4)}, Avg Deep Loss: ${(epochLossDeep/numBatches).toFixed(4)}`);
        }
        
        this.isTraining = false;
        document.getElementById('train').disabled = false;
        document.getElementById('test').disabled = false;
        
        this.updateStatus('Training completed! Click "Test" to see recommendations.');
        this.visualizeEmbeddings();
    }
    
    updateLossChart() {
        const canvas = document.getElementById('lossChart');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (this.lossHistory.length === 0) return;
        
        const maxLoss = Math.max(...this.lossHistory);
        const minLoss = Math.min(...this.lossHistory);
        const range = maxLoss - minLoss || 1;
        
        ctx.strokeStyle = '#2980b9';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        this.lossHistory.forEach((loss, index) => {
            const x = (index / this.lossHistory.length) * canvas.width;
            const y = canvas.height - ((loss - minLoss) / range) * canvas.height;
            index === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();
        
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.fillText(`Min: ${minLoss.toFixed(4)}`, 10, canvas.height - 10);
        ctx.fillText(`Max: ${maxLoss.toFixed(4)}`, 10, 20);
    }
    
    async visualizeEmbeddings() {
        if (!this.deepModel) return;
        this.updateStatus('Computing embedding visualization from deep model...');
        const canvas = document.getElementById('embeddingChart');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        try {
            const sampleSize = Math.min(500, this.itemMap.size);
            const sampleIndices = Array.from({length: sampleSize}, (_, i) => Math.floor(i * (this.itemMap.size -1) / (sampleSize-1) ));
            const sampleItemIds = sampleIndices.map(i => this.reverseItemMap.get(i));
            const sampleItemFeatures = sampleItemIds.map(iid => this.items.get(iid).genres);

            const embeddingsTensor = await this.deepModel.getItemEmbeddings(sampleIndices, sampleItemFeatures);
            const embeddings = embeddingsTensor.arraySync();
            tf.dispose(embeddingsTensor);
            
            const projected = this.computePCA(embeddings, 2);
            
            const xs = projected.map(p => p[0]);
            const ys = projected.map(p => p[1]);
            const xMin = Math.min(...xs), xMax = Math.max(...xs);
            const yMin = Math.min(...ys), yMax = Math.max(...ys);
            const xRange = xMax - xMin || 1;
            const yRange = yMax - yMin || 1;
            
            ctx.fillStyle = 'rgba(41, 128, 185, 0.7)';
            projected.forEach((p, i) => {
                const x = ((p[0] - xMin) / xRange) * (canvas.width - 40) + 20;
                const y = ((p[1] - yMin) / yRange) * (canvas.height - 40) + 20;
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, 2 * Math.PI);
                ctx.fill();
            });
            
            this.updateStatus('Embedding visualization completed.');
        } catch (error) {
            this.updateStatus(`Error in visualization: ${error.message}`);
        }
    }
    
    computePCA(data, dimensions = 2) {
        return tf.tidy(() => {
            const tensor = tf.tensor2d(data);
            const centered = tensor.sub(tensor.mean(0));
            const { _, s, v } = tf.linalg.svd(centered);
            const projected = centered.matMul(v.slice([0, 0], [-1, dimensions]));
            return projected.arraySync();
        });
    }

    async getRecommendations(model, userId, userIndex, ratedItemIds) {
        // Generic recommendation function for any model
        const userEmb = await tf.tidy(async () => {
             if (model instanceof DeepTwoTowerModel) {
                const userFeatures = this.getUserFeatures(userId);
                return model.userForward([userIndex], [userFeatures]);
             } else {
                return model.userForward([userIndex]);
             }
        });

        // Get all item embeddings from the respective model
        const allItemIndices = Array.from(this.itemMap.values());
        const allItemIds = allItemIndices.map(i => this.reverseItemMap.get(i));
        const allItemEmbeddings = await tf.tidy(async () => {
            if (model instanceof DeepTwoTowerModel) {
                 const allItemFeatures = allItemIds.map(iid => this.items.get(iid).genres);
                 return model.itemForward(allItemIndices, allItemFeatures);
            } else {
                return model.itemForward(allItemIndices);
            }
        });
        
        const scores = tf.tidy(() => {
            // Dot product between single user embedding and all item embeddings
            const userRepresentation = userEmb.squeeze();
            return tf.matMul(allItemEmbeddings, userRepresentation.expandDims(1)).squeeze();
        });

        const scoreData = await scores.data();
        tf.dispose([userEmb, allItemEmbeddings, scores]);

        const candidates = [];
        scoreData.forEach((score, itemIndex) => {
            const itemId = this.reverseItemMap.get(itemIndex);
            if (!ratedItemIds.has(itemId)) {
                candidates.push({ itemId, score });
            }
        });
        
        candidates.sort((a, b) => b.score - a.score);
        return candidates.slice(0, 10);
    }
    
    async test() {
        if (!this.simpleModel || !this.deepModel || this.qualifiedUsers.length === 0) {
            this.updateStatus('Models not trained or no qualified users found.');
            return;
        }
        
        this.updateStatus('Generating recommendations from both models...');
        
        try {
            const randomUserId = this.qualifiedUsers[Math.floor(Math.random() * this.qualifiedUsers.length)];
            const userIndex = this.userMap.get(randomUserId);
            const userInteractions = this.userTopRated.get(randomUserId);
            const ratedItemIds = new Set(userInteractions.map(i => i.itemId));
            
            // Get recommendations from both models
            const recsSimple = await this.getRecommendations(this.simpleModel, randomUserId, userIndex, ratedItemIds);
            const recsDeep = await this.getRecommendations(this.deepModel, randomUserId, userIndex, ratedItemIds);
            
            this.displayResults(randomUserId, userInteractions, recsSimple, recsDeep);
            this.updateStatus(`Recommendations generated for User ${randomUserId}.`);
            
        } catch (error) {
            console.error(error);
            this.updateStatus(`Error generating recommendations: ${error.message}`);
        }
    }
    
    displayResults(userId, userInteractions, recsSimple, recsDeep) {
        const resultsDiv = document.getElementById('results');
        const topRated = userInteractions.slice(0, 10);
        
        const createTable = (title, data, isRec = false) => {
            let tableHtml = `
                <div>
                    <h3>${title}</h3>
                    <table>
                        <thead><tr><th>Rank</th><th>Movie</th><th>${isRec ? 'Score' : 'Rating'}</th><th>Year</th></tr></thead>
                        <tbody>`;
            data.forEach((item, index) => {
                const movie = this.items.get(isRec ? item.itemId : item.itemId);
                const value = isRec ? item.score.toFixed(4) : item.rating;
                tableHtml += `<tr><td>${index + 1}</td><td>${movie.title}</td><td>${value}</td><td>${movie.year || 'N/A'}</td></tr>`;
            });
            return tableHtml + '</tbody></table></div>';
        };

        resultsDiv.innerHTML = `
            <h2>Comparison for User ${userId}</h2>
            <div class="results-grid">
                ${createTable('Top 10 Rated (Historical)', topRated)}
                ${createTable('Top 10 Recs (Simple Model)', recsSimple, true)}
                ${createTable('Top 10 Recs (Deep Model)', recsDeep, true)}
            </div>`;
    }
    
    updateStatus(message) {
        document.getElementById('status').textContent = message;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new MovieLensApp();
});
