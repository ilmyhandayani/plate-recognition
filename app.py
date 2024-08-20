#Import Library 
from copyreg import dispatch_table
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from local_utils import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json
import glob
import matplotlib.gridspec as gridspec
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import numpy as np

#inisialisasi flask 
app = Flask(__name__)
app.secret_key = "Plate-Detection"
model = ""

UPLOAD_FOLDER = 'web/static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

IMAGE_FOLDER = 'static/dataset'
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

app.config['IMAGE_FOLDER'] = 'static/dataset'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

#mengatur rute pada tampilan web
@app.route('/', methods=['GET']) #get: http request
def index():
  return render_template('index.html')

@app.route("/dataset", methods=["GET"])
def dataset():
    return render_template('dataset.html')

@app.route('/training')
def training():
    return render_template('training.html', display_training='none')

@app.route('/train', methods=['GET','POST']) #get: http request
def train():
  if request.method == "POST":
      epochs = request.form['epochs']
      EPOCHS = int(epochs)
  dataset_paths = glob.glob("karakter/**/*.png")
  X, y = prep_train(dataset_paths)
  print("dah prep")
  trainX, testX, trainY, testY = split_data (X,y)  
  print("dah split") 
  result, epc = train(trainX, testX, trainY, testY, y, EPOCHS)
  print("dah train") 
  path = visualize(result, EPOCHS)
  EPOCHS = str(EPOCHS)
  return render_template('training.html', visualize="/static/visualize/"+EPOCHS+".png", epoch=epc, display_training='block')

@app.route('/detection', methods=['GET']) #get: http request
def detection():
  return render_template('detection.html', display_testing='none')

@app.route('/upload', methods=['GET', 'POST']) #get & post: http request
def upload():
  if request.method == 'POST':
      f = request.files['file']
      path = "static/testing/"+f.filename
      f.save(path)
      # hasil = main_program(path, f.filename)
      option = request.form['color']
      if option == 'plat_hitam':
            try :
                hasil = main_program_black(path, f.filename)
            except AssertionError as e:
                hasil = "Plate tidak terdeteksi"
                return render_template('detection.html', hasil=hasil, gambar_input="static/testing/"+f.filename, bounding_box="static/box/"+f.filename, plate="static/plate/"+f.filename, preprocessing="static/prep/"+f.filename, segmentasi="static/seg/"+f.filename, display_testing='block')
      elif option == 'plat_putih':
            try :
                hasil = main_program_white(path, f.filename)
            except AssertionError as e:
                hasil = "Plate tidak terdeteksi"
                return render_template('detection.html', hasil=hasil, gambar_input="static/testing/"+f.filename, bounding_box="static/box/"+f.filename, plate="static/plate/"+f.filename, preprocessing="static/prep/"+f.filename, segmentasi="static/seg/"+f.filename, display_testing='block')
      elif option == 'lainnya':
            try:
                hasil = main_program_black(path, f.filename)
            except AssertionError as e:
                try:
                    hasil = main_program_black(path, f.filename)
                except AssertionError as e:
                    hasil = "Plate tidak terdeteksi"
                    return render_template('detection.html', hasil=hasil, gambar_input="static/testing/"+f.filename, bounding_box="static/box/"+f.filename, plate="static/plate/"+f.filename, preprocessing="static/prep/"+f.filename, segmentasi="static/seg/"+f.filename, display_testing='block')

      return render_template('detection.html', hasil=hasil, gambar_input="static/testing/"+f.filename, bounding_box="static/box/"+f.filename, plate="static/plate/"+f.filename, preprocessing="static/prep/"+f.filename, segmentasi="static/seg/"+f.filename, display_testing='block')

@app.route('/detect_all', methods=['GET']) #get: http request
def detect_all():
  return render_template('detect_all.html', display_all='none')

@app.route('/detectfolder', methods=['POST'])
def detectfolder():
    results_folder = "static/uploads" 
    os.makedirs(results_folder, exist_ok=True) 
    txt_path = "static/txt/file.txt"
    with open(txt_path, 'w') as file:
        column1 = "No"
        column2 = "Hasil Pengenalan"
        column3 = "Groundtruth"
        column4 = "Akurasi"
        file.write(f"{column1:4} {column2:20} {column3:20} {column4:10}\n")
    accuracies = []
    uploaded_folder = request.files.getlist('image') 
    for idx, file in enumerate(uploaded_folder, start=1): 
        file_path = os.path.join(results_folder, file.filename)
        print(results_folder) 
        os.makedirs(os.path.dirname(file_path), exist_ok=True) 
        file.save(file_path)
        nama = splitext(basename(file_path))[0][3:12]
        nama = str(nama)
        try:
            hasil = main_program_white(file_path, nama)
            accuracy = calculate_accuracy(hasil, nama)
            print(accuracy)
            # Menambahkan nilai akurasi ke dalam list
            accuracies.append(accuracy)
        except AssertionError as e:
            hasil = "Tidak Terdeteksi"
            accuracy = calculate_accuracy(hasil, nama)
            print(accuracy)
            # Menambahkan nilai akurasi ke dalam list
            accuracies.append(accuracy)
        print(hasil)
        with open(txt_path, 'a') as file:
            file.write("{:4} {:20} {:20} {:.2f}%\n".format(idx, hasil, nama, accuracy))
    average_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    with open(txt_path, 'a') as file:
        file.write(f"\nAkurasi Total = {average_accuracy:.2f}%")
    return render_template('detect_all.html', display_all='block')

@app.route('/download_txt')
def download_txt():
    # Assuming txt_file_path is the path to your generated TXT file
    txt_file_path = 'static/txt/file.txt'
    return send_file(txt_file_path, as_attachment=True)


@app.route('/crud')
def crud():
    image_names = os.listdir(app.config['IMAGE_FOLDER'])
    image_paths = [os.path.join(app.config['IMAGE_FOLDER'], name) for name in image_names]
    return render_template('crud.html', image_data=zip(image_paths, image_names))

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'imageFile' in request.files:
        image_file = request.files['imageFile']

        if image_file.filename != '':
            image_path = os.path.join(app.config['IMAGE_FOLDER'], image_file.filename)
            image_file.save(image_path)
            flash('Image added successfully', 'success')
        else:
            flash('No image selected', 'error')

    return redirect(url_for('crud'))

@app.route('/rename_image/<old_name>', methods=['POST'])
def rename_image(old_name):
    new_name = request.form.get('newName')

    if new_name and old_name in os.listdir(app.config['IMAGE_FOLDER']):
        old_path = os.path.join(app.config['IMAGE_FOLDER'], old_name)
        new_path = os.path.join(app.config['IMAGE_FOLDER'], new_name + ".jpg")

        os.rename(old_path, new_path)
        flash('Image renamed successfully', 'success')
    else:
        flash('Invalid input or image not found', 'error')

    return redirect(url_for('crud'))

@app.route('/delete_image/<name>')
def delete_image(name):
    if name in os.listdir(app.config['IMAGE_FOLDER']):
        image_path = os.path.join(app.config['IMAGE_FOLDER'], name)
        os.remove(image_path)
        flash('Image deleted successfully', 'success')
    else:
        flash('Image not found', 'error')

    return redirect(url_for('crud'))

#TRAINING
def prep_train(dataset_paths):
    # Arange input data and corresponding labels
    X=[]
    labels=[]

    for image_path in dataset_paths:
        label = image_path.split(os.path.sep)[-2]
        image=load_img(image_path,target_size=(80,80))
        image=img_to_array(image)

        X.append(image)
        labels.append(label)

    X = np.array(X,dtype="float16")
    labels = np.array(labels)

    print("[INFO] Find {:d} images with {:d} classes".format(len(X),len(set(labels))))


    # perform one-hot encoding on the labels
    lb = LabelEncoder()
    lb.fit(labels)
    labels = lb.transform(labels)
    y = to_categorical(labels)

    # save label file so we can use in another script
    np.save('license_character_classes.npy', lb.classes_)
    return X, y

def split_data(X,y):
    # split 10% of data as validation set
    (trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)
    return trainX, testX, trainY, testY

def create_model(lr=1e-4,decay=1e-4/25, training=False,output_shape=0):
    baseModel = MobileNetV2(weights="imagenet", 
                            include_top=False,
                            input_tensor=Input(shape=(80, 80, 3)))

    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(3, 3))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(output_shape, activation="softmax")(headModel)
    
    model = Model(inputs=baseModel.input, outputs=headModel)
    
    if training:
        # define trainable lalyer
        for layer in baseModel.layers:
            layer.trainable = True
        # compile model
        optimizer = Adam(lr=lr, decay = decay)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer,metrics=["accuracy"])    
        
    return model

def train(trainX, testX, trainY, testY, y,EPOCHS):
    # data augumentation
    image_gen = ImageDataGenerator(rotation_range=10,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                shear_range=0.1,
                                zoom_range=0.1,
                                fill_mode="nearest"
                                )
    # initilaize initial hyperparameter
    INIT_LR = 1e-4
    model = create_model(lr=INIT_LR, decay=INIT_LR/EPOCHS,training=True,output_shape=y.shape[1])
    BATCH_SIZE = 64

    my_checkpointer = [EarlyStopping(monitor='val_loss', patience=5, verbose=0),
                    ModelCheckpoint(filepath="License_character_recognition.h5", verbose=1, save_weights_only=True)
                    ]

    result = model.fit(image_gen.flow(trainX, trainY, batch_size=BATCH_SIZE), 
                    steps_per_epoch=len(trainX) // BATCH_SIZE, 
                    validation_data=(testX, testY), 
                    validation_steps=len(testX) // BATCH_SIZE, 
                    epochs=EPOCHS, callbacks=my_checkpointer)

    epc = []
    for i in range(len(result.history['loss'])):
      epc.append(f"Epoch {i+1}/15\n29/29 [==============================] - loss: {result.history['loss'][i]:.4f} - accuracy: {result.history['accuracy'][i]:.4f} - val_loss: {result.history['val_loss'][i]:.4f} - val_accuracy: {result.history['val_accuracy'][i]:.4f}")

    return result, epc

def visualize(result, epoch):
    fig = plt.figure(figsize=(14,5))
    grid=gridspec.GridSpec(ncols=2,nrows=1,figure=fig)
    fig.add_subplot(grid[0])
    plt.plot(result.history['accuracy'], label='training accuracy')
    plt.plot(result.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()

    fig.add_subplot(grid[1])
    plt.plot(result.history['loss'], label='training loss')
    plt.plot(result.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    path = 'static/visualize/'+str(epoch)
    plt.savefig(path,dpi=300)
    return path

#DETECTION
def load_model_wpodnet(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)

def load_model_mb(mdl, weight, label):
    json_file = open(mdl, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(weight)
    print("[INFO] Model loaded successfully...")

    labels = LabelEncoder()
    labels.classes_ = np.load(label)
    print("[INFO] Labels loaded successfully...")
    return model, labels

def preprocess_image_white(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

def preprocess_image_black(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.bitwise_not(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

def get_plate_white(image_path, wpod_net, Dmax=608, Dmin = 608):
    vehicle = preprocess_image_white(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return vehicle, LpImg, cor

def get_plate_black(image_path, wpod_net, Dmax=608, Dmin=256):
    vehicle = preprocess_image_black(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return vehicle, LpImg, cor

def draw_box(image_path, cor, thickness=3): 
    pts=[]  
    x_coordinates=cor[0][0]
    y_coordinates=cor[0][1]
    # store the top-left, top-right, bottom-left, bottom-right 
    # of the plate license respectively
    for i in range(4):
        pts.append([int(x_coordinates[i]),int(y_coordinates[i])])
                
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1,1,2))
    vehicle_image = preprocess_image_white(image_path)
                
    cv2.polylines(vehicle_image,[pts],True,(0,255,0),thickness)
    return vehicle_image

def img_processing(LpImg, nama):
    if (len(LpImg)): #check if there is at least one license image
        # Scales, calculates absolute values, and converts the result to 8-bit.
        plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
        
        # convert to grayscale and blur the image
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        
        # Applied inversed thresh_binary 
        binary = cv2.threshold(gray, 180, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE,kernel3)

        # visualize results    
        fig = plt.figure(figsize=(12,7))
        plt.rcParams.update({"font.size":18})
        grid = gridspec.GridSpec(ncols=2,nrows=3,figure = fig)
        plot_image = [plate_image, gray, binary, thre_mor]
        plot_name = ["plate_image","gray","binary", "dilation"]

        for i in range(len(plot_image)):
            fig.add_subplot(grid[i])
            plt.axis(False)
            plt.title(plot_name[i])
            if i ==0:
                plt.imshow(plot_image[i])
            else:
                plt.imshow(plot_image[i],cmap="gray")
        prep = 'static/prep/'+nama
        plt.savefig(prep)
    return thre_mor, plate_image

def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                key=lambda b: b[1][i], reverse=reverse))
    return cnts

def seg_con (thre_mor, plate_image):
    cont, _  = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # creat a copy version "test_roi" of plat_image to draw bounding box
    test_roi = plate_image.copy()

    # Initialize a list which will be used to append charater image
    crop_characters = []

    # define standard width and height of character
    digit_w, digit_h = 30, 60
    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w
        if 1<=ratio<=3.5: # Only select contour with defined ratio
            if h/plate_image.shape[0]>=0.3: # Select contour which has the height larger than 50% of the plate
                # Draw bounding box arroung digit number
                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)

                # Sperate number and gibe prediction
                curr_num = thre_mor[y:y+h,x:x+w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_characters.append(curr_num)

    print("Detect {} letters...".format(len(crop_characters)))
    return test_roi, crop_characters

def predict_from_model(image,model,labels):
                image = cv2.resize(image,(80,80))
                image = np.stack((image,)*3, axis=-1)
                prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
                return prediction

def main_program_white(path, nama) :
    wpod_net_path = "wpod-net.json"
    mdl_path = "MobileNets_character_recognition.json"
    weights_path = "License_character_recognition_weight.h5"
    labels_path = "license_character_classes.npy"
    wpod_net = load_model_wpodnet(wpod_net_path)
    model, labels = load_model_mb(mdl_path, weights_path, labels_path)
    test_image_path = path
    vehicle, LpImg,cor = get_plate_white(test_image_path, wpod_net)

    # fig = plt.figure(figsize=(12,6))
    # grid = gridspec.GridSpec(ncols=2,nrows=1,figure=fig)
    # fig.add_subplot(grid[0])
    # plt.axis(False)
    # plt.imshow(vehicle)
    # grid = gridspec.GridSpec(ncols=2,nrows=1,figure=fig)
    # fig.add_subplot(grid[1])
    plt.figure(figsize=(8,8))
    plt.axis(False)
    plt.imshow(LpImg[0])
    plate = 'static/plate/'+nama
    plt.savefig(plate)

    plt.figure(figsize=(8,8))
    plt.axis(False)
    plt.imshow(draw_box(test_image_path,cor))
    box = 'static/box/'+nama
    plt.savefig(box)
    # plt.savefig(bounding_box, bbox_inches="tight", pad_inches=0,dpi=300)

    thre_mor, plate_image = img_processing(LpImg, nama)
    test_roi, crop_characters = seg_con(thre_mor, plate_image)
    fig = plt.figure(figsize=(10,6))
    plt.axis(False)
    plt.imshow(test_roi)
    seg = 'static/seg/'+nama
    plt.savefig(seg)
    # plt.savefig(output_path_segbon, bbox_inches="tight", pad_inches=0,dpi=300)
    fig = plt.figure(figsize=(14,4))
    grid = gridspec.GridSpec(ncols=len(crop_characters),nrows=1,figure=fig)
    for i in range(len(crop_characters)):
        fig.add_subplot(grid[i])
        plt.axis(False)
        plt.imshow(crop_characters[i],cmap="gray")
        
    # plt.savefig(output_path_seg, bbox_inches="tight", pad_inches=0)
    fig = plt.figure(figsize=(15,3))
    cols = len(crop_characters)
    grid = gridspec.GridSpec(ncols=cols,nrows=1,figure=fig)

    final_string = ''  # Initialize an empty string to store characters

    for i, character in enumerate(crop_characters):
        fig.add_subplot(grid[i])
        title = np.array2string(predict_from_model(character, model, labels))
        plt.title('{}'.format(title.strip("'[]"), fontsize=20))
        final_string += title.strip("'[]")
        plt.axis(False)
        plt.imshow(character, cmap='gray')

    # Move the return statement outside the loop
    return f'{final_string}'

def main_program_black(path, nama) :
    wpod_net_path = "wpod-net.json"
    mdl_path = "MobileNets_character_recognition.json"
    weights_path = "License_character_recognition_weight.h5"
    labels_path = "license_character_classes.npy"
    wpod_net = load_model_wpodnet(wpod_net_path)
    model, labels = load_model_mb(mdl_path, weights_path, labels_path)
    test_image_path = path
    vehicle, LpImg,cor = get_plate_black(test_image_path, wpod_net)

    # fig = plt.figure(figsize=(12,6))
    # grid = gridspec.GridSpec(ncols=2,nrows=1,figure=fig)
    # fig.add_subplot(grid[0])
    # plt.axis(False)
    # plt.imshow(vehicle)
    # grid = gridspec.GridSpec(ncols=2,nrows=1,figure=fig)
    # fig.add_subplot(grid[1])
    plt.figure(figsize=(8,8))
    plt.axis(False)
    plt.imshow(LpImg[0])
    plate = 'static/plate/'+nama
    plt.savefig(plate)

    plt.figure(figsize=(8,8))
    plt.axis(False)
    plt.imshow(draw_box(test_image_path,cor))
    box = 'static/box/'+nama
    plt.savefig(box)
    # plt.savefig(output_path_bon, bbox_inches="tight", pad_inches=0,dpi=300)

    thre_mor, plate_image = img_processing(LpImg, nama)
    test_roi, crop_characters = seg_con(thre_mor, plate_image)
    fig = plt.figure(figsize=(10,6))
    plt.axis(False)
    plt.imshow(test_roi)
    seg = 'static/seg/'+nama
    plt.savefig(seg)
    # plt.savefig(output_path_segbon, bbox_inches="tight", pad_inches=0,dpi=300)
    fig = plt.figure(figsize=(14,4))
    grid = gridspec.GridSpec(ncols=len(crop_characters),nrows=1,figure=fig)
    for i in range(len(crop_characters)):
        fig.add_subplot(grid[i])
        plt.axis(False)
        plt.imshow(crop_characters[i],cmap="gray")
        
    # plt.savefig(output_path_seg, bbox_inches="tight", pad_inches=0)
    fig = plt.figure(figsize=(15,3))
    cols = len(crop_characters)
    grid = gridspec.GridSpec(ncols=cols,nrows=1,figure=fig)

    final_string = ''  # Initialize an empty string to store characters

    for i, character in enumerate(crop_characters):
        fig.add_subplot(grid[i])
        title = np.array2string(predict_from_model(character, model, labels))
        plt.title('{}'.format(title.strip("'[]"), fontsize=20))
        final_string += title.strip("'[]")
        plt.axis(False)
        plt.imshow(character, cmap='gray')

    # Move the return statement outside the loop
    return f'{final_string}'

#DETECTION ALL
# def calculate_accuracy(true_label, predicted_label):
#     total_characters = len(true_label)
#     correct_predictions = sum(1 for true, pred in zip(true_label, predicted_label) if true == pred)
#     accuracy = (correct_predictions / total_characters) * 100
    
#     return accuracy

def calculate_accuracy(true_label, predicted_label):
    accuracy = 0
    if predicted_label == "Tidak Terdeteksi":
        accuracy = 0
    else :
        total_characters = len(true_label)
        correct_predictions = sum(1 for true, pred in zip(true_label, predicted_label) if true == pred)
        accuracy = (correct_predictions / total_characters) * 100
    print(accuracy)
    return accuracy


app.run(host='127.0.0.1', port='2024', debug=True)