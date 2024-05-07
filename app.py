from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField
from flask_wtf import FlaskForm
from PIL import Image
from pathlib import Path
import os
import matplotlib.pyplot as plt
import librosa
import numpy as np
from pydub import AudioSegment
from flask import Flask, flash, request, redirect, render_template
import matplotlib
import keras
matplotlib.use('Agg')


import librosa
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet import preprocess_input

import numpy as np
import librosa.display, os
import matplotlib.pyplot as plt




app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
ALLOWED_EXTENSIONS = {'wav'}
app.config['UPLOAD_FOLDER'] = 'sample/files'
MAX_FILE_SIZE = 5 * 1024 * 1024
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
uploads_dir = os.path.join(app.instance_path, 'uploads')
audio_dir = os.path.join(app.instance_path, 'audio_clips')
spectro_dir = os.path.join(app.instance_path, 'spectrograms')
os.makedirs(audio_dir, exist_ok=True)
os.makedirs(spectro_dir, exist_ok=True)
os.makedirs(uploads_dir, exist_ok=True)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_STATIC = os.path.join(APP_ROOT, 'static')
APP_SPEC = os.path.join(APP_STATIC, 'std_psw_spec')


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadFileForm()
    if form.validate_on_submit():

        file = form.file.data  # First grab the file

        if 'file' not in request.files:
            flash("No file found")
            return redirect('/')

        file.save(os.path.join(uploads_dir, secure_filename(file.filename)))

        filename = secure_filename(file.filename)

        userFile = os.path.join(uploads_dir, filename)
        print(userFile)
        result = process_audio(userFile)
        print(result)
        print("File received :", filename)
        return render_template("result.html", result=result)
    else:
        return render_template('index.html', form=form)


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

#----------------------Create Spectrogram--------------------
def create_spectrogram(audio_file, image_file):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if not os.path.exists(audio_file):
        print(f"Error: File {audio_file} not found.")
        return

    y, sr = librosa.load(audio_file)
    print("Loaded audio file shape:", y.shape)
    print("Sampling rate:", sr)

    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)

    if not os.path.exists(os.path.dirname(image_file)):
        os.makedirs(os.path.dirname(image_file))
        print("folder created")

    fig.savefig(image_file)
    plt.close(fig)

# ---------------------Create pngs from wavs-----------------------
def create_pngs_from_wavs(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("folder created")

    dir = os.listdir(input_path)

    for i, file in enumerate(dir):
        input_file = os.path.join(input_path, file)
        output_file = os.path.join(output_path, file.replace('.wav', '.png'))
        create_spectrogram(input_file, output_file)

#----------------------Process_audio----------------------
def process_audio(audio_path):
    # Load the audio file using PyDub
    audio = AudioSegment.from_wav(audio_path)

    # Split the audio into 20-second clips
    clip_duration = 10 * 1000  # 10 seconds in milliseconds
    clips = [audio[i:i+clip_duration]
             for i in range(0, len(audio), clip_duration)]
    print(clips)
    for i, clip in enumerate(clips):
        clip_path = os.path.join(audio_dir, f'clip_{i}.wav')
        clip.export(clip_path, format='wav')
        print(clip)
    create_fold_spectrograms()

    spec_list = os.listdir(spectro_dir)
    for spec in spec_list:
        # img_path=Path("static\\spectrograms\\clip_0.png")
        spectro_path = os.path.join(spectro_dir, spec)
        # test_spec_img = Image.open(spectro_path).convert("L")
        # test_array = np.array(test_spec_img)

        # std_spec_img = Image.open(os.path.join(APP_SPEC, 'psw_12.png')).convert("L")
        # std_array = np.array(std_spec_img)
        # print("Shape of test array : ", np.shape(test_array))
        # print("Shape of std array : ", np.shape(std_array))
        # print("ok till array creation")
        #res = cosine_similarity(test_array, std_array)
        # print("Cosine Similarity :", res*100)
        # print('Audio processing completed.')
        # if (res >= 0.9):
        #     delete_files(audio_dir)
        #     delete_files(spectro_dir)
        #     delete_files(uploads_dir)
        #     return "PSW Present"

        #-------------------------------------------------
        model1=load_model("model.keras")
        base_model1 = load_model("base_model.keras")
        print(librosa.__version__)
        # create_spectrogram(audio_path, 'static/Spectrograms/sample1_psw.png')

        x = image.load_img(spectro_path, target_size=(224, 224))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x)
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        y = base_model1.predict(x)
        predictions = model1.predict(y)
        class_labels = ['nopsw', 'psw']

        for i, label in enumerate(class_labels):
            print(f'{label}: {predictions[0][i]}')
            if label=="psw" and predictions[0][i] == 1.0:
                return "PSW Present"


    delete_files(audio_dir)
    delete_files(spectro_dir)
    delete_files(uploads_dir)
    return "PSW Not Present"

    




# ---------------------Cosine Similarity-----------------------


def cosine_similarity(x, y):
    # Ensure length of x and y are the same
    if len(x) != len(y):
        return None

        # Compute the dot product between x and y
    x = x.astype(np.ulonglong)
    y = y.astype(np.ulonglong)
    dot_product = np.sum(x*y)

    # Compute the L2 norms (magnitudes) of x and y
    magnitude_x = np.sum(x*x)
    magnitude_x = np.sqrt(magnitude_x)

    magnitude_y = np.sum(y*y)
    magnitude_y = np.sqrt(magnitude_y)

    # Compute the cosine similarity
    cosine_similarity = dot_product / (magnitude_x * magnitude_y)

    return cosine_similarity

# --------------- Creating Spectrograms----------------------------------------


def create_fold_spectrograms():
    print(audio_dir)
    for audio_file in os.listdir(audio_dir):
        temp_audio_file = os.path.join(audio_dir, audio_file)
        # samples, sample_rate = librosa.load(temp_audio_file)
        # fig = plt.figure()
        # ax = fig.add_subplot(1,1,1)
        # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        # ax.axes.get_xaxis().set_visible(False)
        # ax.axes.get_yaxis().set_visible(False)
        # ax.set_frame_on(False)
        
        filename = os.path.join(spectro_dir, Path(audio_file).name.replace('.wav', '.png'))
        # print(filename)
        # S = librosa.feature.melspectrogram(y=samples, sr=sample_rate)
        # librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
        # plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
        # plt.close('all')
        # plt.close(fig)
        create_spectrogram(temp_audio_file,filename)


# --------DELETE FILE FUNC---------
def delete_files(folder_path):
    # Define the folder path
    # folder_path = 'path/to/your/folder'

    # Get the list of files in the folder
    file_list = os.listdir(folder_path)

    # Iterate over the files and delete them
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        os.remove(file_path)

# ---------------Audio Processing and Output ----------------------------------


def process_audio_old(audio_path):
    # Load the audio file using PyDub
    audio = AudioSegment.from_wav(audio_path)

    # Split the audio into 20-second clips
    clip_duration = 20 * 1000  # 20 seconds in milliseconds
    clips = [audio[i:i+clip_duration]
             for i in range(0, len(audio), clip_duration)]
    print(clips)
    for i, clip in enumerate(clips):
        clip_path = os.path.join(audio_dir, f'clip_{i}.wav')
        clip.export(clip_path, format='wav')
        print(clip)

    create_fold_spectrograms()

    spec_list = os.listdir(spectro_dir)
    for spec in spec_list:
        # img_path=Path("static\\spectrograms\\clip_0.png")
        spectro_path = os.path.join(spectro_dir, spec)
        test_spec_img = Image.open(spectro_path).convert("L")
        test_array = np.array(test_spec_img)

        std_spec_img = Image.open(os.path.join(APP_SPEC, 'psw_12.png')).convert("L")
        std_array = np.array(std_spec_img)
        print("Shape of test array : ", np.shape(test_array))
        print("Shape of std array : ", np.shape(std_array))
        # print("ok till array creation")
        res = cosine_similarity(test_array, std_array)
        print("Cosine Similarity :", res*100)
        print('Audio processing completed.')
        if (res >= 0.9):
            delete_files(audio_dir)
            delete_files(spectro_dir)
            delete_files(uploads_dir)
            return "PSW Present"

    delete_files(audio_dir)
    delete_files(spectro_dir)
    delete_files(uploads_dir)
    return "PSW Not Present"


@app.route('/future')
def future():
    return render_template('future.html')


if __name__ == '__main__':
    app.run()