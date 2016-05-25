using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace garbage
{
    class Program
    {
        static void Main(string[] args)
        {
            var x = new MnistDataLoader();
            var network = new Network(new List<int> { 784, 30, 10});
            network.StochasticGradientDescent(x.TrainingData, 30, 30, 3, x.TestingData);
        }
    }
    public class BigEndianBinaryReader : BinaryReader
    {
        private byte[] a16 = new byte[2];
        private byte[] a32 = new byte[4];
        private byte[] a64 = new byte[8];
        public BigEndianBinaryReader(System.IO.Stream stream) : base(stream) { }
        public override int ReadInt32()
        {
            a32 = base.ReadBytes(4);
            Array.Reverse(a32);
            return BitConverter.ToInt32(a32, 0);
        }
        public override Int16 ReadInt16()
        {
            a16 = base.ReadBytes(2);
            Array.Reverse(a16);
            return BitConverter.ToInt16(a16, 0);
        }
        public override Int64 ReadInt64()
        {
            a64 = base.ReadBytes(8);
            Array.Reverse(a64);
            return BitConverter.ToInt64(a64, 0);
        }
        public override UInt32 ReadUInt32()
        {
            a32 = base.ReadBytes(4);
            Array.Reverse(a32);
            return BitConverter.ToUInt32(a32, 0);
        }

    }
    public class MnistDataLoader
    {
        string root = @"C:\Users\alex\";
        public readonly List<Network.DataSet> TrainingData;
        public readonly List<Network.DataSet> TestingData;
        public MnistDataLoader()
        {
            var trainDataName = "train-images-idx3-ubyte.gz";
            var trainLabelName = "train-labels-idx1-ubyte.gz";
            var testDataName = "t10k-images-idx3-ubyte.gz";
            var testLabelName = "t10k-labels-idx1-ubyte.gz";
            TrainingData = GetDataFromNames(root + trainDataName, root + trainLabelName);
            TestingData = GetDataFromNames(root + testDataName, root + testLabelName);
        }

        public List<Network.DataSet> GetDataFromNames(string dataName, string labelName)
        {
            var datastream = new BigEndianBinaryReader(new GZipStream(File.OpenRead(dataName), CompressionMode.Decompress));
            var labelstream = new BigEndianBinaryReader(new GZipStream(File.OpenRead(labelName), CompressionMode.Decompress));
            if(datastream.ReadInt32() != 2051 || labelstream.ReadInt32() != 2049) throw new Exception("magic number mismatch");
            var size = datastream.ReadInt32();
            var otherSize = labelstream.ReadInt32();
            if(size != otherSize) throw new Exception("Size mismatch between data and labels");
            var results = new List<Network.DataSet>(size);
            var rows = datastream.ReadInt32();
            var columns = datastream.ReadInt32();
            for (var i = 0; i < size; i++)
            {
                var label = labelstream.ReadByte();
                var data = CreateVector.Dense<double>(rows * columns);
                for(var j = 0; j < rows; j++)
                    for (var k = 0; k < columns; k++)
                        data[j * columns + k] = datastream.ReadByte();
                results.Add(new Network.DataSet(data, label));
            }
            return results;
        }
    }

    public class Network
    {
        private List<Matrix<double>> weights; // each layer of the network has a matrix of weights
        private List<Vector<double>> biases; // each layer of the network has a vector of biases
        private List<int> sizes; // each layer of the network has a size (number of nodes)

        public Network(List<int> sizes)
        {
            this.sizes = sizes;
            // randomly initialize biases and weights
            biases = sizes.Skip(1).Select(y => CreateVector.Random<double>(y)).ToList();
            weights = sizes.Take(sizes.Count - 1).Zip(sizes.Skip(1), (x, y) => CreateMatrix.Random<double>(y, x)).ToList();
        }

        public Vector<double> Sigmoid(Vector<double> z)
        {
            return CreateVector.Dense(z.Select(SpecialFunctions.Logistic).ToArray());
            //return 1.0 / (1.0 + z.Negate().PointwiseExp());
        }

        public Vector<double> Feedforward(Vector<double> a) // a is nx1 but not a vector for some reason
        {
            foreach (var x in biases.Zip(weights, (bias, weight) => new {b = bias, w = weight}))
                a = Sigmoid(x.w * a + x.b);
            return a;
        }

        public struct DataSet
        {
            public readonly Vector<double> Data;
            public readonly int Label;

            public DataSet(Vector<double> data, int label)
            {
                Data = data;
                Label = label;
            }
        }

        public void StochasticGradientDescent(List<DataSet> trainingData, int epochs, int miniBatchSize, double eta, List<DataSet> testData = null)
        {
            var nTest = testData?.Count ?? 0;
            var n = trainingData.Count;
            foreach (var x in xrange(0, epochs))
            {
                var shuffled = trainingData.OrderBy(a => Guid.NewGuid()).ToList();
                var miniBatches = xrange(0, n, miniBatchSize).Select(k => shuffled.Skip(k).Take(miniBatchSize).ToList()).ToList();
                foreach (var b in miniBatches)
                {
                    updateMiniBatch(b, eta);
                }
                if (testData != null)
                {
                    Console.WriteLine($"Epoch {x}: {evaluate(testData)} / {nTest}");
                }
                else
                {
                    Console.WriteLine($"Epoch {x} complete");
                }
            }
        }

        private int evaluate(List<DataSet> testData)
        {
            var x = testData.Select(a => new { res = Feedforward(a.Data).MaximumIndex(), label = a.Label });
            return x.Count(a => a.res == a.label);
        }

        private void updateMiniBatch(List<DataSet> batch, double eta)
        {
            var nabla_b = biases.Select(b => CreateVector.Dense<double>(b.Count)).ToList();
            var nabla_w = weights.Select(w => CreateMatrix.Dense<double>(w.RowCount, w.ColumnCount)).ToList();
            var bag = new ConcurrentBag<Tuple<List<Vector<double>>, List<Matrix<double>>>>();
            Parallel.ForEach(batch, x => 
            //foreach (var x in batch)
            { 
                bag.Add(backprop(x));
            });
            foreach (var delta_nabla in bag)
            {
                nabla_b = nabla_b.Zip(delta_nabla.Item1, (nb, dnb) => nb + dnb).ToList();
                nabla_w = nabla_w.Zip(delta_nabla.Item2, (nw, dnw) => nw + dnw).ToList();
            }
            weights = weights.Zip(nabla_w, (w, nw) => w - ((eta / batch.Count) * nw)).ToList();
            biases = biases.Zip(nabla_b, (b, nb) => b - ((eta / batch.Count) * nb)).ToList();
        }

        private Tuple<List<Vector<double>>, List<Matrix<double>>> backprop(DataSet dataSet)
        {
            var nabla_b = biases.Select(b => CreateVector.Dense<double>(b.Count)).ToList();
            var nabla_w = weights.Select(w => CreateMatrix.Dense<double>(w.RowCount, w.ColumnCount)).ToList();

            // feed forward
            var activation = dataSet.Data;
            var activations = new List<Vector<double>>();
            activations.Add(activation);
            var zs = new List<Vector<double>>();
            foreach (var x in biases.Zip(weights, (b, w) => new { b = b, w = w }))
            {
                var z = x.w * activation + x.b;
                zs.Add(z);
                activation = Sigmoid(z);
                activations.Add(activation);
            }

            // backward pass
            var delta = CostDerivative(activations[activations.Count - 1], dataSet.Label).PointwiseMultiply(SigmoidPrime(zs[zs.Count - 1]));
            nabla_b[nabla_b.Count - 1] = delta;

            var deltaMatrix = CreateMatrix.Dense<double>(1, delta.Count);
            deltaMatrix.SetRow(0, delta);
            var activationMatrix = CreateMatrix.Dense<double>(1, activations[activations.Count - 2].Count);
            activationMatrix.SetRow(0, activations[activations.Count - 2]);
            nabla_w[nabla_w.Count - 1] = deltaMatrix.Transpose() * activationMatrix;

            // additional backward passes
            foreach (var L in xrange(2, sizes.Count))
            {
                var z = zs[zs.Count - L];
                var sp = SigmoidPrime(z);
                delta = (weights[weights.Count - L + 1].Transpose() * delta).PointwiseMultiply(sp);
                nabla_b[nabla_b.Count - L] = delta;

                deltaMatrix = CreateMatrix.Dense<double>(1, delta.Count);
                deltaMatrix.SetRow(0, delta);
                activationMatrix = CreateMatrix.Dense<double>(1, activations[activations.Count - L - 1].Count);
                activationMatrix.SetRow(0, activations[activations.Count - L - 1]);
                nabla_w[nabla_w.Count - L] = deltaMatrix.Transpose() * activationMatrix;
            }

            return new Tuple<List<Vector<double>>, List<Matrix<double>>>(nabla_b, nabla_w);
        }

        private Vector<double> SigmoidPrime(Vector<double> z)
        {
            return Sigmoid(z).PointwiseMultiply(1 - Sigmoid(z));
        }

        private Vector<double> CostDerivative(Vector<double> activations, int y)
        {
            return activations - y;
        }

        public static IEnumerable<int> xrange(int start, int stop, int step = 1)
        {
            for (int i = start; i < stop; i += step) yield return i;
        }
    }
}
