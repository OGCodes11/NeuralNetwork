import java.io.*;
import java.util.Random;

public class RELUNeuron {
    private double[] weights;
    private double bias;

    public RELUNeuron(int numInputs) {
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

        return Math.max(0.0, sum);
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