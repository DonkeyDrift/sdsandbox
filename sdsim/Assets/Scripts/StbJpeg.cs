using System;
using System.Runtime.InteropServices;

/// <summary>
/// stb_image_write(stb_jpeg)的 P/Invoke 封装——让 JPEG 编码可以在工作线程跑
/// (ImageConversion.EncodeToJPG 限主线程)。dylib 由 tools/stb_jpeg 构建,
/// 放 Assets/Plugins/stb/;缺失时 CameraSensor 回退主线程编码(仍无 ReadPixels 停顿)。
/// </summary>
public static class StbJpeg
{
    [DllImport("stbjpeg")]
    private static extern int stbj_write_jpg(byte[] rgb, int w, int h, int quality, out IntPtr ptr);

    [DllImport("stbjpeg")]
    private static extern void stbj_free(IntPtr p);

    public static bool Available { get; private set; } = true;

    /// <summary>rgb24 编码;失败(缺库/编码错)返回 null。</summary>
    public static byte[] Encode(byte[] rgb, int w, int h, int quality)
    {
        if (!Available)
            return null;
        try
        {
            int len = stbj_write_jpg(rgb, w, h, quality, out IntPtr ptr);
            if (len <= 0 || ptr == IntPtr.Zero)
                return null;
            byte[] jpg = new byte[len];
            Marshal.Copy(ptr, jpg, 0, len);
            stbj_free(ptr);
            return jpg;
        }
        catch (DllNotFoundException)
        {
            Available = false;
            return null;
        }
    }
}
