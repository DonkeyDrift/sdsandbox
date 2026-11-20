// make-dark-icon.swift — Donkey 模拟器图标双版本生成器（2026-11-18，19 日修订）
//
// 输入 = 从 PlayerIcon.icns 提取的原始 1024 图稿（白底小车，直角满方——
// Dock 里看到的圆角是 macOS 给 icns 加的遮罩，Assets.car iconstack 路线
// 不自动加，因此圆角必须烘进图稿本体）。
//
// 输出（写到 <out-dir>）：
//   icon-day.png   浅色版 = 原稿仅加圆角透明遮罩（主体逐像素不动）
//   icon-night.png 深色版 = 边缘泛洪把背景白换纯黑 #000000 + 同一圆角遮罩
//
// 圆角半径取 macOS 应用图标标准遮罩 ≈ 边长 × 0.2237（1024 → 229px）。
// 近白判定：R/G/B 均 ≥225 且与边缘 4 连通（被主体包住的内部白色不动）。
//
// 用法：swift make-dark-icon.swift <raw-light.png> <out-dir>
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

let args = CommandLine.arguments
guard args.count == 3 else {
    FileHandle.standardError.write("usage: make-dark-icon.swift <raw-light.png> <out-dir>\n".data(using: .utf8)!)
    exit(2)
}
let srcURL = URL(fileURLWithPath: args[1])
let outDir = URL(fileURLWithPath: args[2])
try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

guard let src = CGImageSourceCreateWithURL(srcURL as CFURL, nil),
      let rawImage = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
    FileHandle.standardError.write("cannot read \(args[1])\n".data(using: .utf8)!)
    exit(2)
}
let W = rawImage.width, H = rawImage.height
let space = CGColorSpace(name: CGColorSpace.sRGB)!

func makeContext() -> CGContext {
    CGContext(data: nil, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * 4,
              space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
}

// ---- 1) 深色版底图：边缘泛洪近白 → 纯黑 ----
let flood = makeContext()
flood.draw(rawImage, in: CGRect(x: 0, y: 0, width: W, height: H))
guard let base = flood.data else { exit(2) }
let px = base.bindMemory(to: UInt8.self, capacity: W * H * 4)

func isBackground(_ i: Int) -> Bool {
    px[i] >= 225 && px[i + 1] >= 225 && px[i + 2] >= 225
}

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

// ---- 2) 圆角遮罩 + 输出两版 ----
// Dock 实测（2026-11-20）：满幅 .icon 图稿在 Dock 里比标准图标大一圈
//（本机量得 132px vs ZCode 105px）——可见图标须按 Apple 标准安全区
// 占画布 ~80.5%（824/1024）。圆角半径同比例缩放。
let artworkScale: CGFloat = 824.0 / 1024.0

func roundedImage(_ ctx: CGContext) -> CGImage {
    let masked = makeContext()
    let inset = CGFloat(W) * (1.0 - artworkScale) / 2.0
    let radius = CGFloat(W) * 0.2237 * artworkScale
    let rect = CGRect(x: inset, y: inset, width: CGFloat(W) - inset * 2, height: CGFloat(H) - inset * 2)
    let path = CGPath(roundedRect: rect, cornerWidth: radius, cornerHeight: radius, transform: nil)
    masked.addPath(path)
    masked.clip()
    masked.draw(ctx.makeImage()!, in: rect)
    return masked.makeImage()!
}

func write(_ image: CGImage, _ name: String) {
    let url = outDir.appendingPathComponent(name)
    let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, image, nil)
    CGImageDestinationFinalize(dest)
    print("wrote \(url.path)")
}

let light = makeContext()
light.draw(rawImage, in: CGRect(x: 0, y: 0, width: W, height: H))
write(roundedImage(light), "icon-day.png")
write(roundedImage(flood), "icon-night.png")
print("dark fill repainted \(painted) background pixels; corner radius \(Int(CGFloat(W) * 0.2237))px")
