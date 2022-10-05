# package imports
import os
from IPython.utils import io
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Sequential
from keras.applications.imagenet_utils import preprocess_input as pi
from keras.applications.resnet import ResNet50
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.utils import image_utils


# set global variables
UPLOAD_FOLDER = "web_app/static/uploads/"
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "secret key"
ALLOWED_EXTENSIONS = set(["jpg", "jpeg"])


class DetectorAPI:
    """
    This class allows us to load the pretrained Faster RCN Inception V2 COCO model.
    Arguments:
        path_to_ckpt: path to pretrained model's frozen_inference_graph.pb
    Output:
        None
    """

    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v2.io.gfile.GFile(self.path_to_ckpt, "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name("image_tensor:0")
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name(
            "detection_boxes:0"
        )
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name(
            "detection_scores:0"
        )
        self.detection_classes = self.detection_graph.get_tensor_by_name(
            "detection_classes:0"
        )
        self.num_detections = self.detection_graph.get_tensor_by_name(
            "num_detections:0"
        )

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
            [
                self.detection_boxes,
                self.detection_scores,
                self.detection_classes,
                self.num_detections,
            ],
            feed_dict={self.image_tensor: image_np_expanded},
        )

        im_height, im_width, _ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (
                int(boxes[0, i, 0] * im_height),
                int(boxes[0, i, 1] * im_width),
                int(boxes[0, i, 2] * im_height),
                int(boxes[0, i, 3] * im_width),
            )

        return (
            boxes_list,
            scores[0].tolist(),
            [int(x) for x in classes[0].tolist()],
            int(num[0]),
        )

    def close(self):
        self.sess.close()
        self.default_graph.close()


# initiate inception V3 model
inceptionV3_model = Sequential()
inceptionV3_model.add(GlobalAveragePooling2D(input_shape=(5, 5, 2048)))
inceptionV3_model.add(Dense(133, activation="softmax"))
inceptionV3_model.compile(
    loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
)
inceptionV3_model.load_weights("saved_models/weights.best.inceptionv3.hdf5")

# initiate ResNet50 model
ResNet50_model = ResNet50(weights="imagenet")

# initiate pretrained Faster RCN Inception V2 COCO model
inception_V2_model = DetectorAPI(
    path_to_ckpt="faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"
)

# manually initiate dog names from training set
dog_names = [
    "ages/train/001.Affenpinscher",
    "ages/train/002.Afghan_hound",
    "ages/train/003.Airedale_terrier",
    "ages/train/004.Akita",
    "ages/train/005.Alaskan_malamute",
    "ages/train/006.American_eskimo_dog",
    "ages/train/007.American_foxhound",
    "ages/train/008.American_staffordshire_terrier",
    "ages/train/009.American_water_spaniel",
    "ages/train/010.Anatolian_shepherd_dog",
    "ages/train/011.Australian_cattle_dog",
    "ages/train/012.Australian_shepherd",
    "ages/train/013.Australian_terrier",
    "ages/train/014.Basenji",
    "ages/train/015.Basset_hound",
    "ages/train/016.Beagle",
    "ages/train/017.Bearded_collie",
    "ages/train/018.Beauceron",
    "ages/train/019.Bedlington_terrier",
    "ages/train/020.Belgian_malinois",
    "ages/train/021.Belgian_sheepdog",
    "ages/train/022.Belgian_tervuren",
    "ages/train/023.Bernese_mountain_dog",
    "ages/train/024.Bichon_frise",
    "ages/train/025.Black_and_tan_coonhound",
    "ages/train/026.Black_russian_terrier",
    "ages/train/027.Bloodhound",
    "ages/train/028.Bluetick_coonhound",
    "ages/train/029.Border_collie",
    "ages/train/030.Border_terrier",
    "ages/train/031.Borzoi",
    "ages/train/032.Boston_terrier",
    "ages/train/033.Bouvier_des_flandres",
    "ages/train/034.Boxer",
    "ages/train/035.Boykin_spaniel",
    "ages/train/036.Briard",
    "ages/train/037.Brittany",
    "ages/train/038.Brussels_griffon",
    "ages/train/039.Bull_terrier",
    "ages/train/040.Bulldog",
    "ages/train/041.Bullmastiff",
    "ages/train/042.Cairn_terrier",
    "ages/train/043.Canaan_dog",
    "ages/train/044.Cane_corso",
    "ages/train/045.Cardigan_welsh_corgi",
    "ages/train/046.Cavalier_king_charles_spaniel",
    "ages/train/047.Chesapeake_bay_retriever",
    "ages/train/048.Chihuahua",
    "ages/train/049.Chinese_crested",
    "ages/train/050.Chinese_shar-pei",
    "ages/train/051.Chow_chow",
    "ages/train/052.Clumber_spaniel",
    "ages/train/053.Cocker_spaniel",
    "ages/train/054.Collie",
    "ages/train/055.Curly-coated_retriever",
    "ages/train/056.Dachshund",
    "ages/train/057.Dalmatian",
    "ages/train/058.Dandie_dinmont_terrier",
    "ages/train/059.Doberman_pinscher",
    "ages/train/060.Dogue_de_bordeaux",
    "ages/train/061.English_cocker_spaniel",
    "ages/train/062.English_setter",
    "ages/train/063.English_springer_spaniel",
    "ages/train/064.English_toy_spaniel",
    "ages/train/065.Entlebucher_mountain_dog",
    "ages/train/066.Field_spaniel",
    "ages/train/067.Finnish_spitz",
    "ages/train/068.Flat-coated_retriever",
    "ages/train/069.French_bulldog",
    "ages/train/070.German_pinscher",
    "ages/train/071.German_shepherd_dog",
    "ages/train/072.German_shorthaired_pointer",
    "ages/train/073.German_wirehaired_pointer",
    "ages/train/074.Giant_schnauzer",
    "ages/train/075.Glen_of_imaal_terrier",
    "ages/train/076.Golden_retriever",
    "ages/train/077.Gordon_setter",
    "ages/train/078.Great_dane",
    "ages/train/079.Great_pyrenees",
    "ages/train/080.Greater_swiss_mountain_dog",
    "ages/train/081.Greyhound",
    "ages/train/082.Havanese",
    "ages/train/083.Ibizan_hound",
    "ages/train/084.Icelandic_sheepdog",
    "ages/train/085.Irish_red_and_white_setter",
    "ages/train/086.Irish_setter",
    "ages/train/087.Irish_terrier",
    "ages/train/088.Irish_water_spaniel",
    "ages/train/089.Irish_wolfhound",
    "ages/train/090.Italian_greyhound",
    "ages/train/091.Japanese_chin",
    "ages/train/092.Keeshond",
    "ages/train/093.Kerry_blue_terrier",
    "ages/train/094.Komondor",
    "ages/train/095.Kuvasz",
    "ages/train/096.Labrador_retriever",
    "ages/train/097.Lakeland_terrier",
    "ages/train/098.Leonberger",
    "ages/train/099.Lhasa_apso",
    "ages/train/100.Lowchen",
    "ages/train/101.Maltese",
    "ages/train/102.Manchester_terrier",
    "ages/train/103.Mastiff",
    "ages/train/104.Miniature_schnauzer",
    "ages/train/105.Neapolitan_mastiff",
    "ages/train/106.Newfoundland",
    "ages/train/107.Norfolk_terrier",
    "ages/train/108.Norwegian_buhund",
    "ages/train/109.Norwegian_elkhound",
    "ages/train/110.Norwegian_lundehund",
    "ages/train/111.Norwich_terrier",
    "ages/train/112.Nova_scotia_duck_tolling_retriever",
    "ages/train/113.Old_english_sheepdog",
    "ages/train/114.Otterhound",
    "ages/train/115.Papillon",
    "ages/train/116.Parson_russell_terrier",
    "ages/train/117.Pekingese",
    "ages/train/118.Pembroke_welsh_corgi",
    "ages/train/119.Petit_basset_griffon_vendeen",
    "ages/train/120.Pharaoh_hound",
    "ages/train/121.Plott",
    "ages/train/122.Pointer",
    "ages/train/123.Pomeranian",
    "ages/train/124.Poodle",
    "ages/train/125.Portuguese_water_dog",
    "ages/train/126.Saint_bernard",
    "ages/train/127.Silky_terrier",
    "ages/train/128.Smooth_fox_terrier",
    "ages/train/129.Tibetan_mastiff",
    "ages/train/130.Welsh_springer_spaniel",
    "ages/train/131.Wirehaired_pointing_griffon",
    "ages/train/132.Xoloitzcuintli",
    "ages/train/133.Yorkshire_terrier",
]


def check_file_type(filename):
    """
    Checks file type of uploaded file.
    Arguments:
        filename: file name of uploaded file
    Output:
        filename: file name of uploaded file if file is of allowed type
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def ResNet50_predict_labels(img_path):
    """
    Predict labels using the pretrained ResNet50 Imagenet model.
    Arguments:
        img_path: path to image file
    Output:
        Class label with highest predicted probability.
    """
    # returns prediction vector for image located at img_path
    img = pi(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


def dog_detector(img_path):
    """
    Predicts if image shows dog using the pretrained ResNet50 Imagenet model.
    Arguments:
        img_path: path to image file
    Output:
        True if image shows dog, False otherwise.
    """
    prediction = ResNet50_predict_labels(img_path)
    return (prediction <= 268) & (prediction >= 151)


def human_detector(img_path):
    """
    Predicts if image shows human using the pretrained Faster RCN Inception V2 COCO model.
    Arguments:
        img_path: path to image file
    Output:
        True if image shows human, False otherwise.
    """
    threshold = 0.95
    img = plt.imread(img_path)
    boxes, scores, classes, num = inception_V2_model.processFrame(img)
    for i in range(len(boxes)):
        # Class 1 represents human
        if classes[i] == 1 and scores[i] > threshold:
            return True
    return False


def path_to_tensor(img_path):
    """
    Converts an image to a 4D tensor suitable for supplying to a Keras CNN.
    Arguments:
        img_path: path to image file
    Output:
        4D tensor with shape (n, 224, 224, 3), where 'n' is number of images.
    """
    # loads RGB image as PIL.Image.Image type
    img = image_utils.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image_utils.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def extract_InceptionV3(tensor):
    """
    Extracts bottleneck features for pretrained InceptionV3 model.
    Arguments:
        tensor: image as a 4D tensor
    Output:
        bottleneck features for pretrained InceptionV3 model.
    """
    return InceptionV3(weights="imagenet", include_top=False).predict(
        preprocess_input(tensor)
    )


def predict_dog_breed(img_path):
    """
    Predicts the dog breed present in an image.
    Arguments:
        img_path : path to the image file
    Output:
        predicted name/class of dog breed.
    """
    # extract bottleneck features
    bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))

    # get predicted vector
    with io.capture_output() as captured:
        pred_vector = inceptionV3_model.predict(bottleneck_feature)

    # return predicted dog breed
    return dog_names[np.argmax(pred_vector)]


def breed_classification(img_path):
    """
    Twofolded classification task:
    1. decision if a dog or human is detected in the given image.
    2. if dog or human is detected, predict the most probable dog breed for this image.
    Arguments:
        img_path : path to the image file
    Output:
        result string.
    """
    # get detections for both humans and dogs
    with io.capture_output() as captured:
        dog_detected = dog_detector(img_path)
        human_detected = human_detector(img_path)

    # predict if it is a dog
    if dog_detected == 1:
        with io.capture_output() as captured:
            predicted_breed = predict_dog_breed(img_path).partition(".")[-1]
        return ("Hello doggo, you seem to be a: {}").format(predicted_breed)

    # predict if it is a human
    elif human_detected == 1:
        with io.capture_output() as captured:
            predicted_breed = predict_dog_breed(img_path).partition(".")[-1]
        return ("Hello human being, you look a little bit like a: {}").format(
            predicted_breed
        )

    else:
        return "No dogs or humans detected."


@app.route("/")
def upload_form():
    """
    Loads upload_image.html used for user image input.
    Arguments:
        None
    Output:
        'upload.html' template
    """
    return render_template("upload_image.html")


@app.route("/", methods=["POST"])
def upload_image():
    """
    Load the user-selected image.
    Arguments:
        None
    Output:
        'request.html' if file is not uploaded or incorrect file type is uploaded
        'upload.html' with the answer of image classification if the correct file
        type is uploaded.
    """
    if "file" not in request.files:
        flash("Your image has not been uploaded correctly. Please try again.")
        return redirect(request.url)
    file = request.files["file"]
    if file.filename == "":
        flash("You have not selected any image. Please try again.")
        return redirect(request.url)
    if file and check_file_type(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        flash("Your image has been uploaded:")
        result = breed_classification(
            os.path.join(app.config["UPLOAD_FOLDER"], filename)
        )
        flash(result)
        return render_template("upload_image.html", filename=filename)
    else:
        flash("Allowed image types are -> jpg, jpeg")
        return redirect(request.url)


@app.route("/display/<filename>")
def display_image(filename):
    """
    Displays the user-selected and loaded image.
    Arguments:
        filename : name of uploaded file
    Output:
        path of the user-selected and loaded image.
    """
    # print('display_image filename: ' + filename)
    return redirect(url_for("static", filename="uploads/" + filename), code=301)


def main():
    app.run(host="0.0.0.0", port=3000, debug=True)


if __name__ == "__main__":
    main()
