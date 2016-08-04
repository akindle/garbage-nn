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
            var inputLayer = new Layer(784, 100, 0.5, 
                new SoftmaxLayer(100, 10, 0.5));
            
            var x = new MnistDataLoader();
            x.Load().Wait();
            //var network = new Network(new List<int> {784, 300, 30, 10});
            var i = 0;
            while (true)
            {
                Network.SGD(x.TrainingData, 1000, x.TestingData, inputLayer);
                i++;
                Console.Write(".");
                if (i%80 == 0)
                {
                    Console.WriteLine();
                    Console.WriteLine($"{Network.Evaluate(x.TestingData, inputLayer)} / {x.TestingData.Count}");
                    Console.WriteLine($"{Network.Evaluate(x.TrainingData, inputLayer)} / {x.TrainingData.Count}");
                    Console.WriteLine();
                }
                //network.StochasticGradientDescent(x.TrainingData, 300, 3, x.TestingData).Wait();
            }
        }
    }
}