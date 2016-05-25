using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using garbage;
namespace garbageTest
{
    [TestClass]
    public class UnitTest1
    {
        [TestMethod]
        public void TestMethod1()
        {
            var n = new Network(new List<int> {10, 20, 10});
            var d = new MnistDataLoader();
        }
    }
}
