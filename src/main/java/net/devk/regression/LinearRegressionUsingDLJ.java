package net.devk.regression;

import ai.djl.Model;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.ParameterList;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;

public class LinearRegressionUsingDLJ {

    private static final Logger logger = LoggerFactory.getLogger(LinearRegressionUsingDLJ.class);

    private static ArrayDataset loadArray(NDArray features, NDArray labels, int batchSize, boolean shuffle) {
        return new ArrayDataset.Builder()
                .setData(features) // set the features
                .optLabels(labels) // set the labels
                .setSampling(batchSize, shuffle) // set the batch size and random sampling
                .build();
    }


    public static void main(String[] args) throws TranslateException, IOException {

        logger.info("the data set will be created based on the already selected best weights and bias");
        float[] weights = new float[]{2};
        float bias = 3f;
        logger.info("weights = {}", Arrays.toString(weights));
        logger.info("bias = {}", bias);

        int numberOfGeneratedExamples = 1000;
        try (NDManager manager = NDManager.newBaseManager()) {
            logger.info("generating examples...");
            DataPoints dataPoints = DataPoints.syntheticData(manager, weights, bias, numberOfGeneratedExamples);
            NDArray features = dataPoints.getX();
            NDArray labels = dataPoints.getY();

            int batchSize = 10;

            ArrayDataset dataset = loadArray(features, labels, batchSize, false);


            Model model = Model.newInstance("lin-reg");

            SequentialBlock net = new SequentialBlock();
            Linear linearBlock = Linear.builder().optBias(true).setUnits(1).build();
            net.add(linearBlock);

            model.setBlock(net);

            Loss l2loss = Loss.l2Loss();

            float lr = 0.03f;  // Learning Rate
            Tracker lrt = Tracker.fixed(lr);
            Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

            DefaultTrainingConfig config = new DefaultTrainingConfig(l2loss)
                    .optOptimizer(sgd) // Optimizer (loss function)
                    .optDevices(manager.getEngine().getDevices(1)) // single GPU
                    .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

            Trainer trainer = model.newTrainer(config);

            // First axis is batch size - won't impact parameter initialization
            // Second axis is the input size
            trainer.initialize(new Shape(batchSize, 1));

            Metrics metrics = new Metrics();
            trainer.setMetrics(metrics);


            int numEpochs = 3;

            for (int epoch = 1; epoch <= numEpochs; epoch++) {
                System.out.printf("Epoch %d\n", epoch);
                // Iterate over dataset
                for (Batch batch : trainer.iterateDataset(dataset)) {
                    // Update loss and evaulator
                    EasyTrain.trainBatch(trainer, batch);
                    // Update parameters
                    trainer.step();

                    batch.close();
                }
                // reset training and validation evaluators at end of epoch
                trainer.notifyListeners(listener -> listener.onEpoch(trainer));
            }

            Block layer = model.getBlock();
            ParameterList params = layer.getParameters();
            NDArray wParam = params.valueAt(0).getArray();
            NDArray bParam = params.valueAt(1).getArray();

            NDArray trueW = manager.create(weights);

            NDArray estimatedWeight = wParam.reshape(trueW.getShape());
            float estimatedBias = bParam.getFloat();
            logger.info("estimated weight: {}", estimatedWeight);
            logger.info("estimated bias: {}", estimatedBias);
            logger.info("Error in estimating estimatedWeight: {}", String.format("%f", trueW.sub(estimatedWeight).toFloatArray()[0]));
            logger.info("Error in estimating initialBias: {}", String.format("%f", bias - estimatedBias));

//            Path modelDir = Paths.get("../models/lin-reg");
//            Files.createDirectories(modelDir);

//            model.setProperty("Epoch", Integer.toString(numEpochs)); // save epochs trained as metadata

//            model.save(modelDir, "lin-reg");

        }
    }


}
