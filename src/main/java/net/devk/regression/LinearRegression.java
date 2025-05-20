package net.devk.regression;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
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
import java.util.function.Supplier;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

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
            float[] floatArray = firstFeature.toFloatArray();
            ScatterTrace scatterTrace = createScatterTrace(floatArray, labels.toFloatArray());
            Supplier<DoubleStream> ds = () -> IntStream.range(0, floatArray.length)
                    .mapToDouble(i -> floatArray[i]);
            ScatterTrace lineTrace = createLineTrace(ds.get().min().getAsDouble(), ds.get().max().getAsDouble(), weights[indexOfFeatureToPlot], bias);

            plot(lineTrace, scatterTrace);

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


        }
    }
}
