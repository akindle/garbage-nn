using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace garbage
{
    public class BigEndianBinaryReader : BinaryReader
    {
        private byte[] a16 = new byte[2];
        private byte[] a32 = new byte[4];
        private byte[] a64 = new byte[8];
        public BigEndianBinaryReader(System.IO.Stream stream) : base(stream) { }

        public async Task<int> ReadInt32Async()
        {
            await BaseStream.ReadAsync(a32, 0, 4);
            Array.Reverse(a32);
            return BitConverter.ToInt32(a32, 0);
        }

        public async Task ReadBytesAsync(byte[] buffer, int offset, int count)
        {
            await BaseStream.ReadAsync(buffer, offset, count);
        }

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
}