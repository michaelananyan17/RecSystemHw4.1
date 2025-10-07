/**
 * SimpleTwoTowerModel: A basic two-tower model using only embedding lookups.
 * This model learns a latent vector (embedding) for each user and item.
 * Recommendations are based on the dot product similarity between these embeddings.
 */
class SimpleTwoTowerModel {
    constructor(numUsers, numItems, embeddingDim) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embeddingDim = embeddingDim;
        
        // User and item embeddings are the only trainable parameters.
        this.userEmbeddings = tf.variable(tf.randomNormal([numUsers, embeddingDim], 0, 0.05), true, 'user_embeddings_simple');
        this.itemEmbeddings = tf.variable(tf.randomNormal([numItems, embeddingDim], 0, 0.05), true, 'item_embeddings_simple');
        
        this.optimizer = tf.train.adam(0.001);
    }
    
    // User tower: simply looks up the embedding for a given user index.
    userForward(userIndices) {
        return tf.gather(this.userEmbeddings, userIndices);
    }
    
    // Item tower: simply looks up the embedding for a given item index.  
    itemForward(itemIndices) {
        return tf.gather(this.itemEmbeddings, itemIndices);
    }
    
    async trainStep(userIndices, itemIndices) {
        const loss = () => tf.tidy(() => {
            const userEmbs = this.userForward(userIndices);
            const itemEmbs = this.itemForward(itemIndices);
            
            // In-batch sampled softmax loss.
            // For each (user, item) pair in the batch, all other items in the batch are treated as negatives.
            const logits = tf.matMul(userEmbs, itemEmbs, false, true); // Shape: [batch, batch]
            
            // The diagonal of the logits matrix corresponds to the positive (user, item) pairs.
            const labels = tf.oneHot(tf.range(0, userIndices.length, 1, 'int32'), userIndices.length);
            
            return tf.losses.softmaxCrossEntropy(labels, logits);
        });
        
        const { value, grads } = this.optimizer.computeGradients(loss, [this.userEmbeddings, this.itemEmbeddings]);
        this.optimizer.applyGradients(grads);
        tf.dispose(grads);
        
        const lossVal = await value.data();
        tf.dispose(value);
        return lossVal[0];
    }
}


/**
 * DeepTwoTowerModel: An advanced two-tower model using MLPs.
 * This model incorporates user and item features to create richer representations.
 * User Tower: Consumes user ID + user features (age, gender, occupation).
 * Item Tower: Consumes item ID + item features (genres).
 * Both towers are MLPs that output a final embedding.
 */
class DeepTwoTowerModel {
    constructor(numUsers, numItems, embeddingDim, userFeatureDim, itemFeatureDim, hiddenLayers = [64, 32]) {
        // ID Embeddings
        this.userEmbeddings = tf.variable(tf.randomNormal([numUsers, embeddingDim], 0, 0.05), true, 'user_embeddings_deep');
        this.itemEmbeddings = tf.variable(tf.randomNormal([numItems, embeddingDim], 0, 0.05), true, 'item_embeddings_deep');

        // User Tower MLP Layers
        this.userDense1 = tf.layers.dense({ units: hiddenLayers[0], activation: 'relu', inputShape: [embeddingDim + userFeatureDim] });
        this.userDense2 = tf.layers.dense({ units: hiddenLayers[1], activation: null }); // Output layer

        // Item Tower MLP Layers
        this.itemDense1 = tf.layers.dense({ units: hiddenLayers[0], activation: 'relu', inputShape: [embeddingDim + itemFeatureDim] });
        this.itemDense2 = tf.layers.dense({ units: hiddenLayers[1], activation: null }); // Output layer

        this.trainableVars = [
            this.userEmbeddings, this.itemEmbeddings,
            ...this.userDense1.getWeights(), ...this.userDense2.getWeights(),
            ...this.itemDense1.getWeights(), ...this.itemDense2.getWeights()
        ];
        
        this.optimizer = tf.train.adam(0.001);
    }
    
    // User tower: concatenates user ID embedding with features, then passes through an MLP.
    userForward(userIdices, userFeatures) {
        return tf.tidy(() => {
            const idEmbs = tf.gather(this.userEmbeddings, userIdices);
            const featureTensor = tf.tensor2d(userFeatures);
            const combined = tf.concat([idEmbs, featureTensor], 1);
            let output = this.userDense1.apply(combined);
            output = this.userDense2.apply(output);
            return tf.linalg.l2Normalize(output, -1); // Normalize output embedding
        });
    }
    
    // Item tower: concatenates item ID embedding with features, then passes through an MLP.
    itemForward(itemIndices, itemFeatures) {
        return tf.tidy(() => {
            const idEmbs = tf.gather(this.itemEmbeddings, itemIndices);
            const featureTensor = tf.tensor2d(itemFeatures);
            const combined = tf.concat([idEmbs, featureTensor], 1);
            let output = this.itemDense1.apply(combined);
            output = this.itemDense2.apply(output);
            return tf.linalg.l2Normalize(output, -1); // Normalize output embedding
        });
    }

    async trainStep(userIdices, userFeatures, itemIndices, itemFeatures) {
        const loss = () => tf.tidy(() => {
            const userReps = this.userForward(userIdices, userFeatures);
            const itemReps = this.itemForward(itemIndices, itemFeatures);
            
            // In-batch sampled softmax loss, same logic as the simple model.
            const logits = tf.matMul(userReps, itemReps, false, true);
            const labels = tf.oneHot(tf.range(0, userIdices.length, 1, 'int32'), userIdices.length);
            
            return tf.losses.softmaxCrossEntropy(labels, logits);
        });

        const { value, grads } = this.optimizer.computeGradients(loss, this.trainableVars);
        this.optimizer.applyGradients(grads);
        tf.dispose(grads);
        
        const lossVal = await value.data();
        tf.dispose(value);
        return lossVal[0];
    }
    
    // Helper function for visualization, gets final embeddings for a sample of items.
    getItemEmbeddings(itemIndices, itemFeatures) {
        return this.itemForward(itemIndices, itemFeatures);
    }
}
