import cv2
import streamlit as st
from deepface import DeepFace
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Desativar a mensagem de aviso
st.set_option('deprecation.showPyplotGlobalUse', False)

# Melhoria 1: Definir as emoções como um conjunto ordenado
emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
emotions_count = {emotion: 0 for emotion in emotions}

def analyze_expressions(emotion):
    for key in emotion:
        emotions_count[key] += emotion[key]

def report_expressions():
    total = sum(emotions_count.values())
    st.write("##### Facial Expression Report #####")

    # Tabela com as emoções detectadas e suas porcentagens
    table_data = []
    for emotion in emotions:
        percentage = emotions_count[emotion] * 100 / total
        table_data.append([emotion.capitalize(), '{0} %'.format(round(percentage, 2))])
    st.table(table_data)

    if total == 0:
        st.write("Nenhuma emoção detectada na imagem.")
    else:
        # Cálculo da média e desvio padrão das emoções
        mean = total / len(emotions_count)
        std_dev = np.sqrt(sum([(emotions_count[key] - mean)**2 for key in emotions_count.keys()]) / len(emotions_count))

        # Tabela com a média e desvio padrão das emoções
        stats_data = [
            ['Média', '{0} %'.format(round(mean, 2))],
            ['Desvio Padrão', '{0} %'.format(round(std_dev, 2))]
        ]
        st.table(stats_data)

        # Criação da distribuição normal
        x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 100)
        y = norm.pdf(x, mean, std_dev)

        # Plotagem do histograma das emoções e da distribuição normal
        fig, ax = plt.subplots()
        ax.hist(emotions_count.values(), bins=len(emotions_count), density=True, alpha=0.5, label='Emoções detectadas')
        ax.plot(x, y, 'r--', label='Distribuição normal')
        ax.set_xlabel('Emoções')
        ax.set_ylabel('Probabilidade')
        ax.set_title('Distribuição de probabilidade das emoções detectadas')
        ax.legend()
        ax.set_xticklabels(emotions_count.keys(), rotation=45)
        st.pyplot(fig)

# Melhoria 2: Definir uma função para exibir informações de contato
def show_contact_info():
    st.write("---")
    st.write("Desenvolvido por Dr. Dheiver Santos")
    st.write("Telefone: 51 989889898")
    st.write("E-mail: dheiver.santos@gmail.com")
    st.write("Site: dheiver.com.br")

def start_analyzing_mimic(time):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    start_time = st.session_state.time
    end_time = start_time + time
    while st.session_state.time < end_time:
        ret, frame = cap.read()
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if len(results) > 0:
            result_analyzer = results[0]['emotion']
        else:
            result_analyzer = emotions

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_PLAIN
            dominant_expression = max(result_analyzer, key=result_analyzer.get)
            st.write('Dominant Facial Expression {0}'.format(dominant_expression))
            analyze_expressions(result_analyzer)

            cv2.putText(frame, dominant_expression, (50, 50), font, 3, (0, 0, 255), 2, cv2.LINE_4)
            st.image(frame, channels="BGR")

        else:
            st.write("Nenhum rosto detectado na imagem capturada pela webcam.")

        st.session_state.time = st.session_state.time + 1

        if st.session_state.time >= end_time:
            report_expressions()

        if cv2.waitKey(1) & 0xFF == ord('q'):
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
        st.session_state.time = 0
        time = st.sidebar.slider('Tempo de análise (segundos)', 1, 60, 10)
        if st.button('Iniciar análise'):
            start_analyzing_mimic(time)
    elif option == 'Carregar foto':
        # Widget de upload de foto
        uploaded_file = st.file_uploader("Selecione uma imagem", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            # Lê a imagem
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

