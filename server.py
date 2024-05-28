from flask import Flask, render_template,request
import io
from PIL import Image
import torch
from torchvision import transforms
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/workspace')
def workspace():
    return render_template('workspace.html',message=None)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    if file:
        try:
            image = Image.open(io.BytesIO(file.read()))
            res = process_image(image)
            return render_template('workspace.html', message=res), 200
        except Exception as e:
            return str(e), 500
    return 'File upload failed', 400

def process_image(image):
    newmodel = torch.load("model.pth",map_location=torch.device('cpu'))
    newmodel.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    classnames = ['Bacterial leaf blight','Brown spot','Healthy','Leaf blast','Leaf scald','Narrow brown spot']
    with torch.no_grad():
        output = newmodel(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    _, predicted_class = torch.max(probabilities, 0)
    return classnames[predicted_class.item()]

if __name__ == '__main__':
    app.run(debug=True)