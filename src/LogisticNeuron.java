import java.io.*;
import java.util.Random;

public class LogisticNeuron {
    private double[] weights;
    private double bias;
    private double learningRate = 0.01;

    public LogisticNeuron(int numInputs) {
        weights = new double[numInputs];
        Random rand = new Random();
        for (int i = 0; i < numInputs; i++) {
            weights[i] = rand.nextGaussian() * 0.1;
        }
        bias = 0.0;
    }

    public double output(double[] inputs) {
        double sum = bias;
        for (int i = 0; i < inputs.length; i++) {
            sum += weights[i] * inputs[i];
        }

        return 1.0 / (1.0 + Math.exp(-sum));
    }

    public void train(double[] inputs, double target) {
        double prediction = output(inputs);
        double error = target - prediction;

        double derivative = prediction * (1.0 - prediction);

        for (int i = 0; i < weights.length; i++) {
            weights[i] += learningRate * error * derivative * inputs[i];
        }
        bias += learningRate * error * derivative;
    }

    public void write(DataOutputStream out) throws IOException {
        out.writeInt(weights.length);
        for (double w : weights) {
            out.writeDouble(w);
        }
        out.writeDouble(bias);
    }

    public void read(DataInputStream in) throws IOException {
        int length = in.readInt();
        weights = new double[length];
        for (int i = 0; i < length; i++) {
            weights[i] = in.readDouble();
        }
        bias = in.readDouble();
    }
}
