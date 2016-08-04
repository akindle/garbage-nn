using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace garbage
{
    public class Network
    {
        private readonly Random rand = new Random(0);
        private readonly List<int> sizes; // each layer of the network has a size (number of nodes)
        private List<Vector<double>> biases; // each layer of the network has a vector of biases

        private List<Matrix<double>> del_b;
        private List<Matrix<double>> del_w;
        private List<Matrix<double>> weights; // each layer of the network has a matrix of weights 

        public Network(List<int> sizes)
        {
            if (Control.TryUseNativeCUDA())
            {
                Console.WriteLine("CUDA!");
            }
            else if (Control.TryUseNativeMKL())
            {
                Console.WriteLine("MKL!");
            }
            else if (Control.TryUseNativeOpenBLAS())
            {
                Console.WriteLine("OpenBLAS!");
            }
            else if (Control.TryUseNative())
            {
                Console.WriteLine("Some native provider?!");
            }
            else
            {
                Console.WriteLine("No native provider :(");
            }
            this.sizes = sizes;
            // randomly initialize biases and weights
            biases = sizes.Skip(1).Select(y => CreateVector.Random<double>(y, 219)).ToList();
            weights =
                sizes.Take(sizes.Count - 1)
                    .Zip(sizes.Skip(1), (x, y) => CreateMatrix.Random<double>(y, x, 219))
                    .ToList();
        }

        public Vector<double> Sigmoid(Vector<double> z)
        {
            return z.Map(SpecialFunctions.Logistic);
        }

        public Matrix<double> Sigmoid(Matrix<double> z)
        {
            return z.Map(SpecialFunctions.Logistic);
        }

        public async Task StochasticGradientDescent(List<DataSet> trainingData, int miniBatchSize,
            double eta,
            List<DataSet> testData = null)
        {
            await Task.Run(() =>
            {
                var shuffled = trainingData.OrderBy(a => rand.Next());
                var miniBatches =
                    Generate.LinearRange(0, trainingData.Count, miniBatchSize)
                        .Select(k => shuffled.Skip((int) k).Take(miniBatchSize).ToList())
                        .Select(a => new Minibatch(a));

                var stopwatch = Stopwatch.StartNew();
                foreach (var batch in miniBatches)
                {
                    if (del_b == null)
                        del_b = biases.Select(b => CreateMatrix.Dense<double>(b.Count, batch.inputs.Count)).ToList();
                    if (del_w == null)
                        del_w = weights.Select(w => CreateMatrix.Dense<double>(w.RowCount, w.ColumnCount)).ToList();

                    // setup 
                    // first activation set is also the input
                    var activation = batch.inputMatrix;
                    // we want a matrix of biases for matrix mathing things quickly
                    var biasMatrices = new List<Matrix<double>>();
                    foreach (var bias in biases)
                    {
                        var biasMatrix =
                            CreateMatrix.DenseOfColumnVectors(
                                Generate.LinearRange(1, activation.ColumnCount).Select(_ => bias));
                        biasMatrices.Add(biasMatrix);
                    }

                    // feed forward
                    // we need intermediate values for backpropagating and we need them in reverse order, so cache them on stacks
                    var activationMatrices = new Stack<Matrix<double>>();
                    var sigmoidPrimeMatrices = new Stack<Matrix<double>>();
                    activationMatrices.Push(batch.inputMatrix);
                    foreach (var x in biasMatrices.Zip(weights, (b, w) => new {b, w}))
                    {
                        activation = Sigmoid(x.w*activation + x.b);
                        activationMatrices.Push(activation);
                        sigmoidPrimeMatrices.Push(activation.PointwiseMultiply(1 - activation));
                    }

                    // backward passes 
                    Matrix<double> delta = null;
                    for (var L = 1; L < sizes.Count; L++)
                    {
                        var sp = sigmoidPrimeMatrices.Pop();
                        var a1 = activationMatrices.Pop();
                        if (L > 1)
                        {
                            delta = weights[weights.Count - L + 1].TransposeThisAndMultiply(delta).PointwiseMultiply(sp);
                        }
                        else
                        {
                            // initialize delta with the hadamard product of the cost function and the sigmoid prime
                            delta = (a1 - batch.labelMatrix).PointwiseMultiply(sp);
                            a1 = activationMatrices.Pop();
                        }
                        del_b[del_b.Count - L] = delta;
                        del_w[del_w.Count - L] = delta*a1.Transpose();
                    }

                    weights = weights.Zip(del_w, (w, nw) => w - eta/batch.inputs.Count*nw).ToList();
                    biases =
                        biases.Zip(
                            del_b.Select(
                                a => a.ReduceColumns((vector, doubles) => vector + doubles).Divide(a.ColumnCount)),
                            (b, nb) => b - eta/batch.inputs.Count*nb).ToList();
                    //  updateMiniBatch(batch, eta);
                }
                Console.WriteLine($"Minibatch set done in {stopwatch.Elapsed}");
                if (testData != null)
                {
                    stopwatch = Stopwatch.StartNew();
                    Console.WriteLine($"Epoch: {evaluate(testData)} / {testData.Count}");
                    Console.WriteLine($"evaluate done in {stopwatch.Elapsed}");
                }
            });
        }

        private int evaluate(List<DataSet> testData)
        {
            var i = 0;
            var testMatrix = CreateMatrix.DenseOfColumnVectors(testData.Select(a => a.Data));
            var activation = testMatrix;
            var biasMatrices = new List<Matrix<double>>();
            foreach (var bias in biases)
            {
                var biasMatrix =
                    CreateMatrix.DenseOfColumnVectors(
                        Generate.LinearRange(1, activation.ColumnCount).Select(_ => bias));
                biasMatrices.Add(biasMatrix);
            }
            foreach (var x in biasMatrices.Zip(weights, (b, w) => new { b, w }))
            {
                activation = Sigmoid(x.w * activation + x.b);
            }

            var matrixResults = activation.EnumerateColumns().Select(a => a.MaximumIndex());
            var k = matrixResults.Zip(testData.Select(a => a.Label.MaximumIndex()), (a, b) => a == b).Count(a => a);
            return k;
        }

        public class DataSet
        {
            public readonly Vector<double> Data;
            public readonly Vector<double> Label;
            public Vector<double> PredictedLabel;

            public DataSet(Vector<double> data, int label, int labelDimensionality)
            {
                Data = data;
                Label = CreateVector.Dense<double>(labelDimensionality);
                Label[label] = 1;
            }
        }

        public class Minibatch
        {
            public readonly Matrix<double> inputMatrix;
            public readonly List<DataSet> inputs;
            public readonly Matrix<double> labelMatrix;

            public Minibatch(List<DataSet> inputs)
            {
                this.inputs = inputs;
                inputMatrix = CreateMatrix.DenseOfColumnVectors(inputs.Select(a => a.Data));
                labelMatrix = CreateMatrix.DenseOfColumnVectors(inputs.Select(a => a.Label));
            }
        }
    }
}