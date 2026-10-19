import java.io.*;
import java.util.Scanner;

public class TrainNN {
    public static void main(String[] args) throws IOException {
        if (args.length < 2) {
            System.out.println("Usage: java TrainNN <model_file> <training_dir>");
            return;
        }

        String modelFileName = args[0];
        File trainingDir = new File(args[1]);

        int numInputs = 100;
        NeuralNetwork nn = new NeuralNetwork(numInputs);

        File[] files = trainingDir.listFiles();
        if (files == null) return;

        for (File f : files) {
            if (f.isFile()) {
                double[] inputVector = fileToVector(f, numInputs);
                double target = f.getName().toLowerCase().contains("will") ? 1.0 : 0.0;
                nn.train(inputVector, target);
            }
        }

        try (DataOutputStream out = new DataOutputStream(new FileOutputStream(modelFileName))) {
            nn.write(out);
        }
        System.out.println("Training complete. Model saved.");
    }

    public static double[] fileToVector(File file, int size) throws IOException {
        double[] vector = new double[size];
        try (Scanner sc = new Scanner(file)) {
            while (sc.hasNext()) {
                String word = sc.next().toLowerCase().replaceAll("[^a-z]", "");
                if (!word.isEmpty()) {
                    int index = Math.abs(word.hashCode()) % size;
                    vector[index]++;
                }
            }
        }
        return vector;
    }
}