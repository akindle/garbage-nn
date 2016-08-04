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

        public Vector<double> Feedforward(Vector<double> a)
        {
            foreach (var x in biases.Zip(weights, (bias, weight) => new {b = bias, w = weight}))
                a = Sigmoid(x.w*a + x.b);
            return a;
        }

        public async Task StochasticGradientDescent(List<DataSet> trainingData, int epochs, int miniBatchSize,
            double eta,
            List<DataSet> testData = null)
        {
            await Task.Run(() =>
            {
                var nTest = testData?.Count ?? 0;
                var n = trainingData.Count;
                for(var i = 0; i < epochs; i++)
                {
                    var shuffled = trainingData.OrderBy(a => rand.Next());
                    var miniBatches =
                        Generate.LinearRange(0, n, miniBatchSize).Select(k => shuffled.Skip((int)k).Take(miniBatchSize).ToList());
                    // updateMiniBatch(shuffled.Take(miniBatchSize), eta);
                    var stopwatch = Stopwatch.StartNew();
                    foreach (var batch in miniBatches)
                    {
                        updateMiniBatch(new Minibatch(batch), eta);
                        //  updateMiniBatch(batch, eta);
                    }
                    Console.WriteLine($"Minibatch set done in {stopwatch.Elapsed}");
                    if (testData != null)
                    {
                        Console.WriteLine($"Epoch {i}: {evaluate(testData)} / {nTest}");
                    }
                    else
                    {
                        Console.WriteLine($"Epoch {i} complete");
                    }
                }
            });
        }

        private void updateMiniBatch(Minibatch batch, double eta)
        {
            if (del_b == null)
                del_b = biases.Select(b => CreateMatrix.Dense<double>(b.Count, batch.inputs.Count)).ToList();
            if (del_w == null)
                del_w = weights.Select(w => CreateMatrix.Dense<double>(w.RowCount, w.ColumnCount)).ToList();
            backprop(batch, ref del_b, ref del_w);
            weights = weights.Zip(del_w, (w, nw) => w - eta/batch.inputs.Count*nw).ToList();
            biases =
                biases.Zip(
                    del_b.Select(a => a.ReduceColumns((vector, doubles) => vector + doubles).Divide(a.ColumnCount)),
                    (b, nb) => b - eta/batch.inputs.Count*nb).ToList();
        }

        private int evaluate(List<DataSet> testData)
        {
            var i = 0;
            foreach (var x in testData)
            {
                x.PredictedLabel = Feedforward(x.Data);
                i += x.PredictedLabel.MaximumIndex() == x.Label.MaximumIndex() ? 1 : 0;
            }
            var j = testData.Count(a => a.PredictedLabel.MaximumIndex() == a.Label.MaximumIndex());
            return i;
        }

        private void backprop(Minibatch batch, ref List<Matrix<double>> del_b, ref List<Matrix<double>> del_w)
        {
            // feed forward
            var activation = batch.inputMatrix;
            var biasMatrices = new List<Matrix<double>>();
            foreach (var b in biases)
            {
                var bm = CreateMatrix.Dense<double>(b.Count, activation.ColumnCount);
                for (var i = 0; i < bm.ColumnCount; i++) bm.SetColumn(i, b);
                biasMatrices.Add(bm);
            }
            var aStack = new Stack<Matrix<double>>();
            aStack.Push(activation);
            var spStack = new Stack<Matrix<double>>();
            foreach (var x in biasMatrices.Zip(weights, (b, w) => new {b, w}))
            {
                var z = x.w*activation + x.b;
                activation = Sigmoid(z);
                spStack.Push(activation.PointwiseMultiply(1 - activation));
                aStack.Push(activation);
            }

            // backward pass
            // hadamard product of cost derivative and sp
            var delta = (aStack.Pop() - batch.labelMatrix).PointwiseMultiply(spStack.Pop());
            del_b[del_b.Count - 1] = delta;
            del_w[del_w.Count - 1] = delta*aStack.Pop().Transpose();

            // additional backward passes 
            for(var L = 2; L < sizes.Count; L++)
            {
                var sp = spStack.Pop();
                var a = aStack.Pop();

                delta = weights[weights.Count - L + 1].TransposeThisAndMultiply(delta).PointwiseMultiply(sp);
                del_b[del_b.Count - L] = delta;
                del_w[del_w.Count - L] = delta*a.Transpose();
            }

            //return new Tuple<List<Vector<double>>, List<Matrix<double>>>(del_b, del_w);
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