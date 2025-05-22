package net.devk.regression;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.translate.TranslateException;
import tech.tablesaw.api.FloatColumn;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.components.Axis;
import tech.tablesaw.plotly.components.Figure;
import tech.tablesaw.plotly.components.Layout;
import tech.tablesaw.plotly.traces.ScatterTrace;

import java.io.IOException;

public class LinearRegression {


    public static ScatterTrace createLineTrace(double min, double max, float weight, float bias) {
        double[] yLineValues = new double[2];
        yLineValues[0] = weight * min + bias;
        yLineValues[1] = weight * max + bias;
        // Create traces
        return ScatterTrace.builder(new double[]{min, max}, yLineValues)
                .mode(ScatterTrace.Mode.LINE)
                .name(String.format("y = %fx + %f", weight, bias))
                .build();
    }

    public static ScatterTrace createScatterTrace(float[] features, float[] labels) {
        return ScatterTrace.builder(FloatColumn.create("X", features), FloatColumn.create("y", labels))
                .mode(ScatterTrace.Mode.MARKERS)
                .name("Synthetic Data")
                .build();
    }

    public static void plot(ScatterTrace lineTrace, ScatterTrace scatterTrace) {
        // Combine into one figure
        Layout layout = Layout.builder()
                .title("Linear Regression")
                .xAxis(Axis.builder().title("x").build())
                .yAxis(Axis.builder().title("y").build())
                .build();

        Plot.show(new Figure(layout, lineTrace, scatterTrace));
    }


    public static void main(String[] args) throws TranslateException, IOException {

        float[] weights = new float[]{2, -3.4f};
        float bias = 4.2f;
        int indexOfFeatureToPlot = 1;
        int numberOfExamples = 1000;
        try (NDManager manager = NDManager.newBaseManager()) {
            DataPoints dataPoints = DataPoints.syntheticData(manager, weights, bias, numberOfExamples);
            NDArray features = dataPoints.getX();
            NDArray labels = dataPoints.getY();
            NDArray firstFeature = features.get(new NDIndex(String.format(":, %d", indexOfFeatureToPlot)));
            System.out.println("using the first feature of the data points: " + firstFeature);
//            float[] floatArray = firstFeature.toFloatArray();
//            ScatterTrace scatterTrace = createScatterTrace(floatArray, labels.toFloatArray());
//            Supplier<DoubleStream> ds = () -> IntStream.range(0, floatArray.length)
//                    .mapToDouble(i -> floatArray[i]);
//            ScatterTrace lineTrace = createLineTrace(ds.get().min().getAsDouble(), ds.get().max().getAsDouble(), weights[indexOfFeatureToPlot], bias);
//
//            plot(lineTrace, scatterTrace);

            int batchSize = 10;

            ArrayDataset dataset = new ArrayDataset.Builder()
                    .setData(features) // Set the Features
                    .optLabels(labels) // Set the Labels
                    .setSampling(batchSize, false) // set the batch size and random sampling to false
                    .build();

            try (Batch batch = dataset.getData(manager).iterator().next()) {
                // Call head() to get the first NDArray
                NDArray X1 = batch.getData().head();
                NDArray y1 = batch.getLabels().head();
                System.out.println(X1);
                System.out.println(y1);
            }

            // we initialize weights by sampling random numbers from a normal distribution with mean 0 and
            // a standard deviation of  0.01, setting the bias  initialBias to  0
            NDArray randomWeights = manager.randomNormal(0, 0.01f, new Shape(2, 1), DataType.FLOAT32);
            System.out.println("random weights: " + randomWeights);
            NDArray initialBias = manager.zeros(new Shape(1));
            System.out.println("initial bias: " + initialBias);
            NDList params = new NDList(randomWeights, initialBias);


            float lr = 0.03f;  // Learning Rate
            int numEpochs = 10;  // Number of Iterations

            // Attach Gradients
            for (NDArray param : params) {
                param.setRequiresGradient(true);
            }

            for (int epoch = 0; epoch < numEpochs; epoch++) {
                // Assuming the number of examples can be divided by the batch size, all
                // the examples in the training dataset are used once in one epoch
                // iteration. The features and tags of minibatch examples are given by X
                // and y respectively.
                for (Batch batch : dataset.getData(manager)) {
                    NDArray X = batch.getData().head();
                    NDArray y = batch.getLabels().head();

                    try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                        // Minibatch loss in X and y
                        NDArray l = squaredLoss(linreg(X, params.get(0), params.get(1)), y);
                        gc.backward(l);  // Compute gradient on l with respect to estimatedWeight and initialBias
                    }
                    sgd(params, lr, batchSize);  // Update parameters using their gradient

                    batch.close();
                }
                NDArray trainL = squaredLoss(linreg(features, params.get(0), params.get(1)), labels);
                System.out.printf("epoch %d, loss %f\n", epoch + 1, trainL.mean().getFloat());

            }

            NDArray trueW = manager.create(new float[]{2, -3.4f});
            float trueB = 4.2f;

            float[] estimatedWeight = trueW.sub(params.get(0).reshape(trueW.getShape())).toFloatArray();
            System.out.println(String.format("Error in estimating estimatedWeight: [%f, %f]", estimatedWeight[0], estimatedWeight[1]));
            System.out.println(String.format("Error in estimating initialBias: %f", trueB - params.get(1).getFloat()));


        }
    }

    public static NDArray linreg(NDArray X, NDArray w, NDArray b) {
        return X.dot(w).add(b);
    }

    // we need to transform the true value y into the predicted value’s shape yHat.
    // The result returned by the following function will also be the same as the yHat shape.
    public static NDArray squaredLoss(NDArray yHat, NDArray y) {
        return (yHat.sub(y.reshape(yHat.getShape()))).mul
                ((yHat.sub(y.reshape(yHat.getShape())))).div(2);
    }

    // stochastic gradient descent
    public static void sgd(NDList params, float lr, int batchSize) {
        for (int i = 0; i < params.size(); i++) {
            NDArray param = params.get(i);
            // Update param
            // param = param - param.gradient * lr / batchSize
            param.subi(param.getGradient().mul(lr).div(batchSize));
        }
    }

    //TODO what is least squares for estimating the slope?
    //TODO Sum of the squared residuals ( a type of loss function)
    //TODO derivative? https://www.youtube.com/watch?v=sDv4f4s2SB8&t=627s
}
