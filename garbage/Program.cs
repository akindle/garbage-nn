using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
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
            var x = new MnistDataLoader();
            var network = new Network(new List<int> { 784, 100, 30, 10});
            network.StochasticGradientDescent(x.TrainingData, 30000, 3000, 4, x.TestingData);
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
                for (var j = 0; j < (rows * columns); j++)
                {
                    data[j] = datastream.ReadByte();
                }
                results.Add(new Network.DataSet(data, label, 10));
            }
            return results;
        }
    }

    public class Network
    {
        private List<Matrix<double>> weights; // each layer of the network has a matrix of weights
        private List<Matrix<double>> weightsT; // each layer of the network has a matrix of weights
        private List<Vector<double>> biases; // each layer of the network has a vector of biases
        private List<int> sizes; // each layer of the network has a size (number of nodes)

        public Network(List<int> sizes)
        {
            this.sizes = sizes;
            // randomly initialize biases and weights
            biases = sizes.Skip(1).Select(y => CreateVector.Random<double>(y, 219)).ToList();
            weights = sizes.Take(sizes.Count - 1).Zip(sizes.Skip(1), (x, y) => CreateMatrix.Random<double>(y, x, 219)).ToList();
            weightsT = weights.Select(a => a.Transpose()).ToList();
        }

        public Vector<double> Sigmoid(Vector<double> z)
        {
            return CreateVector.Dense(z.Select(SpecialFunctions.Logistic).ToArray());
        }

        public Vector<double> Feedforward(Vector<double> a)
        {
            foreach (var x in biases.Zip(weights, (bias, weight) => new {b = bias, w = weight}))
                a = Sigmoid(x.w * a + x.b);
            return a;
        }

        public class DataSet
        {
            public readonly Vector<double> Data;
            public readonly Vector<double> Label;

            public DataSet(Vector<double> data, int label, int labelDimensionality)
            {
                Data = data;
                Label = CreateVector.Dense<double>(labelDimensionality);
                this.Label[label] = 1;
            }
        }
        private readonly Random rand = new Random(0);
        public void StochasticGradientDescent(List<DataSet> trainingData, int epochs, int miniBatchSize, double eta, List<DataSet> testData = null)
        { 
            var nTest = testData?.Count ?? 0;
            var n = trainingData.Count;
            foreach (var x in xrange(0, epochs))
            {
                var shuffled = trainingData.OrderBy(a => rand.Next());
                var miniBatches = xrange(0, n, miniBatchSize).Select(k => shuffled.Skip(k).Take(miniBatchSize).ToList());
                // updateMiniBatch(shuffled.Take(miniBatchSize), eta);
                var stopwatch = Stopwatch.StartNew();
                foreach (var batch in miniBatches)
                {
                    updateMiniBatch(batch, eta);
                }
                Console.WriteLine($"Minibatch set done in {stopwatch.Elapsed}");
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
            var x = testData.Select(a => new { res = Feedforward(a.Data).MaximumIndex(), label = a.Label.MaximumIndex() });
            return x.Count(a => a.res == a.label);
        }

        private void updateMiniBatch(IEnumerable<DataSet> batch, double eta)
        {
            var del_b = biases.Select(b => CreateVector.Dense<double>(b.Count)).ToList();
            var del_w = weights.Select(w => CreateMatrix.Dense<double>(w.RowCount, w.ColumnCount)).ToList();
            var del_bp = biases.Select(b => CreateVector.Dense<double>(b.Count)).ToList();
            var del_wp = weights.Select(w => CreateMatrix.Dense<double>(w.RowCount, w.ColumnCount)).ToList();
            var batchSize = 0;
            //Parallel.ForEach(batch, x => 
            foreach (var x in batch)
            {
                batchSize++;
                backprop(x, ref del_bp, ref del_wp);
                del_b = del_b.Zip(del_bp, (nb, dnb) => nb + dnb).ToList();
                del_w = del_w.Zip(del_wp, (nw, dnw) => nw + dnw).ToList();
            }
            weights = weights.Zip(del_w, (w, nw) => w - ((eta / batchSize) * nw)).ToList();
            weightsT = weights.Select(a => a.Transpose()).ToList();
            biases = biases.Zip(del_b, (b, nb) => b - ((eta / batchSize) * nb)).ToList();
        }

        private void backprop(DataSet dataSet, ref List<Vector<double>> del_b, ref List<Matrix<double>> del_w)
        {
            // feed forward
            var activation = dataSet.Data;
            var aStack = new Stack<Vector<double>>(); 
            aStack.Push(activation);
            var spStack = new Stack<Vector<double>>();
            foreach (var x in biases.Zip(weights, (b, w) => new { b = b, w = w }))
            {
                var z = x.w * activation + x.b;  
                activation = Sigmoid(z);
                spStack.Push(activation.PointwiseMultiply(1 - activation));
                aStack.Push(activation); 
            }

            // backward pass
            var delta = CostDerivative(aStack.Pop(), dataSet.Label).PointwiseMultiply(spStack.Pop());
            del_b[del_b.Count - 1] = delta;
            del_w[del_w.Count - 1] = delta.ToColumnMatrix() * aStack.Pop().ToRowMatrix();

            // additional backward passes
            foreach (var L in xrange(2, sizes.Count))
            { 
                var sp = spStack.Pop();
                delta = (weightsT[weights.Count - L + 1] * delta).PointwiseMultiply(sp);
                del_b[del_b.Count - L] = delta;
                del_w[del_w.Count - L] = delta.ToColumnMatrix() * aStack.Pop().ToRowMatrix();
            }

            //return new Tuple<List<Vector<double>>, List<Matrix<double>>>(del_b, del_w);
        }

        private Vector<double> SigmoidPrime(Vector<double> z)
        {
            return Sigmoid(z).PointwiseMultiply(1 - Sigmoid(z));
        }

        private Vector<double> CostDerivative(Vector<double> activations, Vector<double> y)
        {
            return activations - y;
        }

        public static IEnumerable<int> xrange(int start, int stop, int step = 1)
        {
            for (int i = start; i < stop; i += step) yield return i;
        }
    }
}
