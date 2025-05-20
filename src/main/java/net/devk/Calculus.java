package net.devk;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import tech.tablesaw.plotly.components.Axis;
import tech.tablesaw.plotly.components.Figure;
import tech.tablesaw.plotly.components.Layout;
import tech.tablesaw.plotly.traces.ScatterTrace;

import java.util.function.Function;

public class Calculus {

    public static void main(String[] args) {
        Function<Double, Double> f = x -> 3 * Math.pow(x, 2) - 4 * x;

        /*
        double h = 0.1;
        for (int i = 0; i < 5; i++) {
            System.out.println("h=" + String.format("%.5f", h) + ", numerical limit="
                    + String.format("%.5f", numericalLim(f, 1, h)));
            h *= 0.1;
        }
         */

        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray X = manager.arange(0f, 3f, 0.1f, DataType.FLOAT64);
            double[] x = X.toDoubleArray();
            double[] fx = new double[x.length];
            for (int i = 0; i < x.length; i++) {
                fx[i] = f.apply(x[i]);
            }


            double[] fg = new double[x.length];
            for (int i = 0; i < x.length; i++) {
                fg[i] = 2 * x[i] - 3;
            }

            Figure figure = plotLineAndSegment(x, fx, fg, "f(x)", "Tangent line(x=1)", "x", "f(x)", 700, 500);
            System.out.println(figure);
        }

    }

    public static Double numericalLim(Function<Double, Double> f, double x, double h) {
        return (f.apply(x + h) - f.apply(x)) / h;
    }

    public static Figure plotLineAndSegment(double[] x, double[] y, double[] segment,
                                            String trace1Name, String trace2Name,
                                            String xLabel, String yLabel,
                                            int width, int height) {
        ScatterTrace trace = ScatterTrace.builder(x, y)
                .mode(ScatterTrace.Mode.LINE)
                .name(trace1Name)
                .build();

        ScatterTrace trace2 = ScatterTrace.builder(x, segment)
                .mode(ScatterTrace.Mode.LINE)
                .name(trace2Name)
                .build();

        Layout layout = Layout.builder()
                .height(height)
                .width(width)
                .showLegend(true)
                .xAxis(Axis.builder().title(xLabel).build())
                .yAxis(Axis.builder().title(yLabel).build())
                .build();

        return new Figure(layout, trace, trace2);
    }

}
