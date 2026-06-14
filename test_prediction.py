from services.predictor import predict_mri

prediction, confidence = predict_mri(
    "sample.jpg"
)

print(prediction)
print(confidence)