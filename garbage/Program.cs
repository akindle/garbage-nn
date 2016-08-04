using System;
using System.Collections.Generic;
using MathNet.Numerics;

namespace garbage
{
    internal class Program
    {
        private static void Main(string[] args)
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
            var inputLayer = new Network.Layer(784, 100, 3, 
                new Network.Layer(100, 10, 3));
            
            var x = new MnistDataLoader();
            x.Load().Wait();
            //var network = new Network(new List<int> {784, 300, 30, 10});
            while (true)
            {
                Console.WriteLine(Network.SGD(x.TrainingData, 3000, x.TestingData, inputLayer));
                //network.StochasticGradientDescent(x.TrainingData, 300, 3, x.TestingData).Wait();
            }
        }
    }
}