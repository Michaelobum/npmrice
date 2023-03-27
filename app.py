import os
from flask import Flask, request, render_template
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

longitud, altura = 150, 150
modelo = './Modelo/modelo.P19'
pesos_modelo = './Modelo/pesos.P19'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    # Guardar la imagen en el servidor
    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Cargar la imagen y hacer la predicción
    img = Image.open(file_path)
    x = img.resize((longitud, altura))
    x = img_to_array(x)
    x = x.reshape((1,) + x.shape)
    array = cnn.predict(x)
    result = array[0]
    answer = int(result.argmax())

    if answer == 0:
        pred = "Acaro"
        recomendacion = "Se Recomienda usar Ovicidas: (Acaristop)"
    elif answer == 1:
        pred = "Barrenador de Tallo"
        recomendacion = "Se Recomienda usar Proclaim5"
    elif answer == 2:
        pred = "Escarabajo"
        recomendacion = "Se Recomienda usar Lexan"
    elif answer == 3:
        pred = "Gusano Cogollero"
        recomendacion = "Se Recomienda usar Gusanol"
    elif answer == 4:
        pred = "Mosca de Sierra"
        recomendacion = "Se Recomienda usar Shermann"
    elif answer == 5:
        pred = "Mosquito"
        recomendacion = "Se Recomienda usar Agroinco"
    elif answer == 6:
        pred = "Pulgon"
        recomendacion = "Se Recomienda usar Aphox"
    elif answer == 7:
        pred = "Saltamones"
        recomendacion = "Se Recomienda usar Venerate"
    elif answer == 8:
        pred = "Cultivo en Excelente estado"
        recomendacion = ""
    os.remove(file_path)

    return render_template('result.html', prediction=pred, recomendacion=recomendacion, image_path=file_path)

if __name__ == '_main_':
    app.run(debug=True)