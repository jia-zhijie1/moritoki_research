public class javatest_american_option_pricing {
    public static void main(String[] args) {
        int n = Integer.parseInt(args[0]);
        double time = Double.parseDouble(args[1]);
        int T = Integer.parseInt(args[2]);
        double alpha = Double.parseDouble(args[3]);
        double sigma = Double.parseDouble(args[4]);
        double x = Double.parseDouble(args[5]);
        double K = Double.parseDouble(args[6]);
        double r = Double.parseDouble(args[7]);

        double Price = AmericanOptionPricing(n, time, T, alpha, sigma, x, K, r);
        System.out.println(Price);

    }

    public static double UpProbability(double alpha, double potential) {
        double p = Math.exp(-alpha * potential) / (Math.exp(alpha * potential) + Math.exp(-alpha * potential));
        return p;
    }

    public static double Delta(int n, int T, double sigma) {
        double dt = (double) T / n;
        double delta = sigma * Math.sqrt(dt);
        return delta;
    }

    public static double AmericanOptionPricing(int n, double time, int T, double alpha, double sigma, double x,
            double K, double r) {
        double delta = Delta(n, T, sigma);
        double dt = (double) T / n;
        double timePlus = time + dt;
        double xPlus = x + delta;
        double xMinus = x - delta;
        double zero = (double) 0;
        double p = UpProbability(alpha, x);

        if (Math.abs(time - T) < dt) {
            return Math.max(zero, x - K);
        } else {
            return Math.max(Math.max(zero, x - K), (p * AmericanOptionPricing(n, timePlus, T, alpha, sigma, xPlus, K, r)
                    + (1 - p) * AmericanOptionPricing(n, timePlus, T, alpha, sigma, xMinus, K, r)) / (1 + r));
        }
    }
}