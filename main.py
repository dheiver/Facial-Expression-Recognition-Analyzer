import cv2
import streamlit as st
from deepface import DeepFace

# Desativar a mensagem de aviso
st.set_option('deprecation.showPyplotGlobalUse', False)

emotions = {
    "angry": 0,
    "disgust": 0,
    "fear": 0,
    "happy": 0,
    "sad": 0,
    "surprise": 0,
    "neutral": 0
}

def analyze_expressions(emotion):
    for key in emotion:
        emotions[key] = emotions[key] + emotion[key];

def report_expressions():
    total = 0;
    st.write("##### Facial Expression Report #####")
    st.write(emotions)
    for key in emotions:
        total += emotions[key];
    for key in emotions:
        percentage = emotions[key] * 100 / total;
        st.write('{0} => % {1}'.format(key, round(percentage, 2)))

def start_analyzing_mimic():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if len(results) > 0:
            result_analyzer = results[0]['emotion']
        else:
            result_analyzer = emotions

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_PLAIN
        st.write('Dominant Facial Expression {0}'.format(max(result_analyzer, key=result_analyzer.get)))
        analyze_expressions(result_analyzer)

        cv2.putText(frame, max(result_analyzer, key=result_analyzer.get), (50, 50), font, 3, (0, 0, 255), 2, cv2.LINE_4)
        st.image(frame, channels="BGR")

        if cv2.waitKey(2) & 0xFF == ord('q'):
            report_expressions()
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    st.title("Facial Expression Recognition & Analyzer")
    st.write("Selecione uma opção abaixo para começar a análise de suas expressões faciais")

    # Widget de seleção de opção: webcam ou upload de foto
    option = st.selectbox('Selecione uma opção', ('Usar webcam', 'Carregar foto'))

    if option == 'Usar webcam':
        # Inicia a análise de expressões faciais usando a webcam
        if st.button('Iniciar análise'):
            start_analyzing_mimic()
    elif option == 'Carregar foto':
        # Widget de upload de foto
        uploaded_file = st.file_uploader("Selecione uma imagem", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            # Lê a imagem carregada
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            # Redimensiona a imagem para o tamanho máximo de 800x800
            max_size = 800
            if image.shape[0] > max_size or image.shape[1] > max_size:
                scale = max_size / max(image.shape[0], image.shape[1])
                image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))

            # Realiza a análise de expressões faciais na imagem carregada
            results = DeepFace.analyze(image, actions=['emotion'])
            emotions = results['emotion']
            st.write('Expressão facial dominante: {0}'.format(max(emotions, key=emotions.get)))
            report_expressions()


if __name__ == '__main__':
    main()
