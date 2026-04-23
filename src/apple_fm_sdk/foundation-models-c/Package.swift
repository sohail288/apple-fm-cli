/*
For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.
*/

// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "foundation-models-c-bindings",
  // FoundationModels / Generable / LanguageModelSession require macOS 26+.
  // Use 26.0 (not 26.4) so the binary is loadable on any macOS 26.x that has the framework.
  platforms: [.macOS("26.0"), .iOS("26.0"), .visionOS("26.0")],
  products: [
    .library(name: "apple_fm_bridge", type: .dynamic, targets: ["FoundationModelsCBindings"]),
    .library(name: "apple_fm_bridge_static", type: .static, targets: ["FoundationModelsCBindings"]),
    .executable(
      name: "fm-c-example",
      targets: ["fm-c-example"]
    )
  ],
  targets: [
    // A placeholder target that exposes the declarations from the bindings header to the bindings library itself.
    .target(
      name: "FoundationModelsCDeclarations"
    ),
    // The main target.
    .target(
      name: "FoundationModelsCBindings",
      dependencies: ["FoundationModelsCDeclarations"],
      publicHeadersPath: "include",
      cSettings: [
        .headerSearchPath("Sources/FoundationModelsCBindings/include")
      ],
      swiftSettings: [
        // This flag enables native async calls and correct property names on macOS 26.4+ SDKs.
        // It is manually flipped for the CI environment or local testing on newer macOS.
        .define("FOUNDATION_MODELS_26_4_API", .when(platforms: [.macOS], configuration: .release))
      ].filter { _ in false } // Placeholder: We will use a more direct approach if this fails.
    ),
    .executableTarget(
      name: "fm-c-example",
      dependencies: [
        .byName(name: "FoundationModelsCBindings")
      ],
      cSettings: [
        .headerSearchPath("../FoundationModelsCBindings/include")
      ]
    ),
    .testTarget(
      name: "FoundationModelsCBindingsTests",
      dependencies: ["FoundationModelsCBindings"]
    )
  ],
  cLanguageStandard: .c99
)
