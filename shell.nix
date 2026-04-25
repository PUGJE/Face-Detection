{ pkgs ? import <nixpkgs> {
    config.permittedInsecurePackages = [
      "python3.13-ecdsa-0.19.1"
      "python3.12-ecdsa-0.19.1"
      "python3.11-ecdsa-0.19.1"
    ];
} }:

let
  pythonEnv = pkgs.python3.withPackages (ps: with ps; [
    # Core Web Framework
    fastapi uvicorn uvloop httptools starlette python-multipart pydantic pydantic-settings anyio h11

    # Database
    sqlalchemy greenlet psycopg2 aiosqlite

    # Supabase
    supabase

    # Authentication & Security
    python-jose passlib bcrypt python-dotenv cryptography cffi pycparser rsa pyasn1 ecdsa

    # Computer Vision & Image Processing
    opencv4 pillow

    # Deep Learning & Face Recognition
    insightface onnx onnxruntime faiss

    # Anti-spoofing helpers
    scikit-image lazy-loader tifffile imageio scikit-learn scipy joblib threadpoolctl

    # Data Processing
    numpy pandas python-dateutil six

    # HTTP utilities
    requests httpx httpcore certifi charset-normalizer idna urllib3

    # Server utilities
    watchfiles websockets click tqdm pyyaml redis

    # Type support
    annotated-types typing-extensions

    # Misc
    easydict flatbuffers protobuf ml-dtypes absl-py
  ]);

in
pkgs.mkShell {
  buildInputs = [
    pythonEnv
    pkgs.nodejs_22 # LTS is safer than 24 for Next.js 16 right now
    
    # Still providing system libraries just in case any underlying C-bindings need them
    pkgs.stdenv.cc.cc.lib
    pkgs.glib
    pkgs.libGL
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
    echo "Nix-native Python environment loaded!"
    echo "You don't need to run pip install anymore."
  '';
}
