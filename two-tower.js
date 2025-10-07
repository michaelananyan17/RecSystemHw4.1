class TwoTowerModel {
    constructor(numUsers, numItems, embeddingDim) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embeddingDim = embeddingDim;
        
        // Initialize embedding tables with small random values
        // Two-tower architecture: separate user and item embeddings
        this.userEmbeddings = tf.variable(
            tf.randomNormal([numUsers, embeddingDim], 0, 0.05), 
            true, 
            'user_embeddings'
        );
        
        this.itemEmbeddings = tf.variable(
            tf.randomNormal([numItems, embeddingDim], 0, 0.05), 
            true, 
            'item_embeddings'
        );
        
        // Adam optimizer for stable training
        this.optimizer = tf.train.adam(0.001);
    }
    
    // User tower: simple embedding lookup
    userForward(userIndices) {
        return tf.gather(this.userEmbeddings, userIndices);
    }
    
    // Item tower: simple embedding lookup  
    itemForward(itemIndices) {
        return tf.gather(this.itemEmbeddings, itemIndices);
    }
    
    // Scoring function: dot product between user and item embeddings
    // Dot product is efficient and commonly used in retrieval systems
    score(userEmbeddings, itemEmbeddings) {
        return tf.sum(tf.mul(userEmbeddings, itemEmbeddings), -1);
    }
    
    async trainStep(userIndices, itemIndices) {
        return await tf.tidy(() => {
            const userTensor = tf.tensor1d(userIndices, 'int32');
            const itemTensor = tf.tensor1d(itemIndices, 'int32');
            
            // In-batch sampled softmax loss:
            // Use all items in batch as negatives for each user
            // Diagonal elements are positive pairs
            const loss = () => {
                const userEmbs = this.userForward(userTensor);
                const itemEmbs = this.itemForward(itemTensor);
                
                // Compute similarity matrix: batch_size x batch_size
                const logits = tf.matMul(userEmbs, itemEmbs, false, true);
                
                // Labels: diagonal elements are positives
                // Use int32 tensor for oneHot indices
                const labels = tf.oneHot(
                    tf.range(0, userIndices.length, 1, 'int32'), 
                    userIndices.length
                );
                
                // Softmax cross entropy loss
                // This encourages positive pairs to have higher scores than negatives
                const loss = tf.losses.softmaxCrossEntropy(labels, logits);
                return loss;
            };
            
            // Compute gradients and update embeddings
            const { value, grads } = this.optimizer.computeGradients(loss);
            
            this.optimizer.applyGradients(grads);
            
            return value.dataSync()[0];
        });
    }
    
    getUserEmbedding(userIndex) {
        return tf.tidy(() => {
            return this.userForward([userIndex]).squeeze();
        });
    }
    
    async getScoresForAllItems(userEmbedding) {
        return await tf.tidy(() => {
            // Compute dot product with all item embeddings
            const scores = tf.dot(this.itemEmbeddings, userEmbedding);
            return scores.dataSync();
        });
    }
    
    getItemEmbeddings() {
        // Return the tensor directly - call arraySync() on the tensor, not this method
        return this.itemEmbeddings;
    }
}
