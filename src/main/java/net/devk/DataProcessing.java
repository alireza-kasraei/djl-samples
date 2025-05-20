package net.devk;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import tech.tablesaw.api.BooleanColumn;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.columns.Column;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

public class DataProcessing {
    public static void main(String[] args) throws IOException {
        File file = new File("../data/");
        file.mkdir();

        String dataFile = "../data/house_tiny.csv";

        // Create file
        File f = new File(dataFile);
        f.createNewFile();

        // Write to file
        try (FileWriter writer = new FileWriter(dataFile)) {
            writer.write("NumRooms,Alley,Price\n"); // Column names
            writer.write("NA,Pave,127500\n");  // Each row represents a data example
            writer.write("2,NA,106000\n");
            writer.write("4,NA,178100\n");
            writer.write("NA,NA,140000\n");
        }

        Table data = Table.read().file("../data/house_tiny.csv");
//        System.out.println(data);
        //To handle missing data, typical methods include imputation and deletion, where imputation replaces missing
        //values with substituted ones, while deletion ignores missing values. Here we will consider imputation.

        Table inputs = data.create(data.columns());
        inputs.removeColumns("Price");
        Table outputs = data.selectColumns("Price");
        System.out.println(inputs);

        Column col = inputs.column("NumRooms");
        System.out.println(col);

        // sets the mean value of the "NumRooms" to the column with the null/empty value
        col.set(col.isMissing(), (int) inputs.nCol("NumRooms").mean());
        System.out.println(inputs);


        StringColumn stringColumn = (StringColumn) inputs.column("Alley");
        List<BooleanColumn> dummies = stringColumn.getDummies();
        inputs.removeColumns(stringColumn);
        inputs.addColumns(DoubleColumn.create("Alley_Pave", dummies.get(0).asDoubleArray()),
                DoubleColumn.create("Alley_nan", dummies.get(1).asDoubleArray())
        );

        try (NDManager nd = NDManager.newBaseManager()) {
            NDArray x = nd.create(inputs.as().doubleMatrix());
            NDArray y = nd.create(outputs.as().intMatrix());
            System.out.println("x= " + x);
            System.out.println("y= " + y);
        }

    }
}
