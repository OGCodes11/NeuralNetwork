import java.io.*;
import java.util.Scanner;

public class UseNN {
    public static void main(String[] args) throws IOException {
        if (args.length < 2) {
            System.out.println("Usage: java UseNN <model_file> <test_file>");
            return;
        }

        int numInputs = 100;
        NeuralNetwork nn = new NeuralNetwork(numInputs);

        try (DataInputStream in = new DataInputStream(new FileInputStream(args[0]))) {
            nn.read(in);
        }

        File testFile = new File(args[1]);
        double totalScore = 0;
        int count = 0;

        try (Scanner sc = new Scanner(testFile)) {
            while (sc.hasNextLine()) {
                String line = sc.nextLine();
                if (line.trim().isEmpty()) continue;

                double[] vector = lineToVector(line, numInputs);
                totalScore += nn.output(vector);
                count++;
            }
        }

        double average = (count > 0) ? totalScore / count : 0.5;
        String author = (average >= 0.5) ? "Will" : "Taylor";
        double confidence = Math.abs(average - 0.5) * 2 * 100;

        System.out.println("Prediction: " + author);
        System.out.printf("Confidence: %.2f%%\n", confidence);
    }

    public static double[] lineToVector(String line, int size) {
        double[] vector = new double[size];
        String[] words = line.toLowerCase().replaceAll("[^a-z ]", "").split("\\s+");
        for (String w : words) {
            if (!w.isEmpty()) {
                int index = Math.abs(w.hashCode()) % size;
                vector[index]++;
            }
        }
        return vector;
    }
}