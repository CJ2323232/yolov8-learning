# Privacy and local image processing

## What the application does

This is a static browser application. Its detection pipeline decodes the image you choose, prepares an input tensor, runs ONNX Runtime Web on your device, and draws predictions in the page.

The app does not send your selected image, filename, image metadata, detection boxes, or prediction scores to an inference server. It does not include analytics, advertising trackers, external inference APIs, or an application account system.

You should choose only images you have permission to use. A public Space does not require visitors to sign in to the application.

## Network requests that still occur

Opening the page makes ordinary web requests to the hosting service. The page downloads its JavaScript/WASM runtime, `helmet_p2.onnx`, and `model_info.json` from the hosting service, including its official download CDN. These downloads transfer the model to your browser; they do not upload your image.

The host may receive information needed to serve requests, such as an IP address, user-agent details, request times, and platform cookies. The application's no-image-upload design is not a promise that the host collects no request logs.

When hosted on Hugging Face, its own platform practices are governed by the [Hugging Face Privacy Policy](https://huggingface.co/privacy). Opening a source or documentation link also makes a request to that destination.

## Browser-side storage and lifetime

Selected files and generated results can remain available to the page while it is open. Closing or reloading the page removes the active application state in the normal browser lifecycle; this is not a secure-erasure guarantee.

The browser may cache downloaded application and model files. Any result that you explicitly save is stored through your browser's download mechanism on your own device. This privacy statement does not describe browser extensions, operating-system features, or files you subsequently share yourself.

## Public package contents

The package contains the application, its trained ONNX model, model information, corresponding project source, and license documents. It does not intentionally include original training photographs, private verification images or tensors, training logs, private account credentials, tokens, personal contact details, or personal computer paths.

The original training checkpoint is not published in this package. The export has separately recorded checksums and privacy checks, described in [MODEL_CARD.md](MODEL_CARD.md). Explicit metadata review is not a formal guarantee against every possible disclosure from learned model parameters.

## Verify the data flow

To inspect this behavior, open your browser's Network panel before selecting an image and running detection. Model/runtime downloads may appear; image files and detection data should not appear as uploads or inference requests. A failed model download should produce a loading error rather than send your image to an external fallback.

Report unexpected behavior through the [project's GitHub issue tracker](https://github.com/CJ2323232/yolov8-learning/issues). Do not attach private images, account details, tokens, or other sensitive material to a public issue.
