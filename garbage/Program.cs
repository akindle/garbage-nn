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
            var network = new Network(new List<int> { 784, 300, 10});
            network.StochasticGradientDescent(x.TrainingData, 30000, 300, 3, x.TestingData);
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
            biases = sizes.Skip(1).Select(y => CreateVector.Random<double>(y, 219)).ToList();
            weights = sizes.Take(sizes.Count - 1).Zip(sizes.Skip(1), (x, y) => CreateMatrix.Random<double>(y, x, 219)).ToList();
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

        public class Minibatch
        {
            public readonly List<DataSet> inputs; 
            public readonly Matrix<double> inputMatrix;
            public readonly Matrix<double> labelMatrix; 

            public Minibatch(List<DataSet> inputs)
            {
                this.inputs = inputs;
                inputMatrix = CreateMatrix.DenseOfColumnVectors(inputs.Select(a => a.Data));
                labelMatrix = CreateMatrix.DenseOfColumnVectors(inputs.Select(a => a.Label));
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
                    updateMiniBatch(new Minibatch(batch), eta);
                  //  updateMiniBatch(batch, eta);
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

        List<Matrix<double>> del_b;
        List<Matrix<double>> del_w;
        private void updateMiniBatch(Minibatch batch, double eta)
        {
            if(del_b == null) del_b = biases.Select(b => CreateMatrix.Dense<double>(b.Count, batch.inputs.Count)).ToList();
            if(del_w == null) del_w = weights.Select(w => CreateMatrix.Dense<double>(w.RowCount, w.ColumnCount)).ToList(); 
            backprop(batch, ref del_b, ref del_w);
            weights = weights.Zip(del_w, (w, nw) => w - eta / batch.inputs.Count * nw).ToList(); 
            biases = biases.Zip(del_b.Select(a => a.ReduceColumns((vector, doubles) => vector + doubles).Divide(a.ColumnCount)), (b, nb) => b - ((eta / batch.inputs.Count) * nb)).ToList();
        }

        private int evaluate(List<DataSet> testData)
        {
            var x = testData.Select(a => new { res = Feedforward(a.Data).MaximumIndex(), label = a.Label.MaximumIndex() });
            return x.Count(a => a.res == a.label);
        }
 
        private void backprop(Minibatch batch, ref List<Matrix<double>> del_b, ref List<Matrix<double>> del_w)
        {
            
            // feed forward
            var activation = batch.inputMatrix;
            var biasMatrices = new List<Matrix<double>>();
            foreach (var b in biases)
            {
                var bm = CreateMatrix.Dense<double>(b.Count, activation.ColumnCount);
                for(var i = 0; i < bm.ColumnCount; i++) bm.SetColumn(i, b);
                biasMatrices.Add(bm);
            }
            var aStack = new Stack<Matrix<double>>();
            aStack.Push(activation);
            var spStack = new Stack<Matrix<double>>();
            foreach (var x in biasMatrices.Zip(weights, (b, w) => new { b = b, w = w }))
            {
                var z = x.w * activation + x.b;
                activation = Sigmoid(z);
                spStack.Push(activation.PointwiseMultiply(1 - activation));
                aStack.Push(activation);
            }

            // backward pass
            var delta = CostDerivative(aStack.Pop(), batch.labelMatrix).PointwiseMultiply(spStack.Pop());
            del_b[del_b.Count - 1] = delta;
            del_w[del_w.Count - 1] = delta * aStack.Pop().Transpose();

            // additional backward passes
            foreach (var L in xrange(2, sizes.Count))
            {
                var sp = spStack.Pop();
                delta = (weights[weights.Count - L + 1].TransposeThisAndMultiply(delta)).PointwiseMultiply(sp);
                del_b[del_b.Count - L] = delta;
                del_w[del_w.Count - L] = delta * aStack.Pop().Transpose();
            }

            //return new Tuple<List<Vector<double>>, List<Matrix<double>>>(del_b, del_w);
        }

        private Matrix<double> CostDerivative(Matrix<double> activations, Matrix<double> labelMatrix)
        {
            return activations - labelMatrix;
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
            if (datastream.ReadInt32() != 2051 || labelstream.ReadInt32() != 2049) throw new Exception("magic number mismatch");
            var size = datastream.ReadInt32();
            var otherSize = labelstream.ReadInt32();
            if (size != otherSize) throw new Exception("Size mismatch between data and labels");
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

}
