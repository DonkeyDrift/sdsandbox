// make-dark-icon.swift — Donkey 模拟器图标深色版生成器（2026-11-18）
//
// 浅色版 = 现行 PlayerIcon 图标本体（白底小车，tools/simulator-icon/icon-light.png，
// 1024×1024 不透明满幅）；深色版要求主体逐像素不动、仅背景变纯黑 #000000。
//
// 做法：从四边向内做 4 连通泛洪，把与边缘连通的近白像素（R/G/B 均 ≥225，
// 即背景白及其过渡）替换为纯黑；被主体包住的内部白色（高光等）不属于
// 背景、保持原样。无外部依赖，macOS 自带 Swift + CoreGraphics 即可复现。
//
// 用法：swift make-dark-icon.swift <light.png> <dark.png>
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

let args = CommandLine.arguments
guard args.count == 3 else {
    FileHandle.standardError.write("usage: make-dark-icon.swift <light.png> <dark.png>\n".data(using: .utf8)!)
    exit(2)
}
let srcURL = URL(fileURLWithPath: args[1])
let dstURL = URL(fileURLWithPath: args[2])

guard let src = CGImageSourceCreateWithURL(srcURL as CFURL, nil),
      let image = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
    FileHandle.standardError.write("cannot read \(args[1])\n".data(using: .utf8)!)
    exit(2)
}
let W = image.width, H = image.height
let space = CGColorSpace(name: CGColorSpace.sRGB)!
let ctx = CGContext(data: nil, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * 4,
                    space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
ctx.draw(image, in: CGRect(x: 0, y: 0, width: W, height: H))
guard let base = ctx.data else { exit(2) }
let px = base.bindMemory(to: UInt8.self, capacity: W * H * 4)

func isBackground(_ i: Int) -> Bool {
    px[i] >= 225 && px[i + 1] >= 225 && px[i + 2] >= 225
}

// 从全部边缘像素出发泛洪（跳过本身就是主体的像素）
var queue = [Int]()
var filled = [Bool](repeating: false, count: W * H)
for x in 0..<W { queue.append(x); queue.append((H - 1) * W + x) }
for y in 0..<H { queue.append(y * W); queue.append(y * W + W - 1) }
var head = 0
var painted = 0
while head < queue.count {
    let p = queue[head]; head += 1
    if p < 0 || p >= W * H || filled[p] { continue }
    let i = p * 4
    if !isBackground(i) { continue }
    filled[p] = true
    px[i] = 0; px[i + 1] = 0; px[i + 2] = 0
    painted += 1
    let x = p % W
    if x > 0 { queue.append(p - 1) }
    if x < W - 1 { queue.append(p + 1) }
    queue.append(p - W)
    queue.append(p + W)
}

guard let out = ctx.makeImage(),
      let dest = CGImageDestinationCreateWithURL(dstURL as CFURL, UTType.png.identifier as CFString, 1, nil) else {
    FileHandle.standardError.write("cannot write \(args[2])\n".data(using: .utf8)!)
    exit(2)
}
CGImageDestinationAddImage(dest, out, nil)
CGImageDestinationFinalize(dest)
print("dark icon written: \(dstURL.path) (\(W)×\(H), background pixels repainted: \(painted))")
