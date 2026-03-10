import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
import os
from datetime import datetime


class DetectorHolistic:
    def __init__(self):
        # Inicializar Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,  # 0, 1 ou 2 (2 é mais preciso, mas mais lento)
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_desenho = mp.solutions.drawing_utils

        # Estilos de desenho personalizados
        self.drawing_spec_pose = self.mp_desenho.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=2)  # Verde
        self.drawing_spec_face = self.mp_desenho.DrawingSpec(
            color=(255, 0, 255), thickness=1, circle_radius=1)  # Rosa
        self.drawing_spec_hands = self.mp_desenho.DrawingSpec(
            color=(255, 255, 0), thickness=2, circle_radius=2)  # Amarelo

    def processar_frame(self, frame):
        """Processa o frame com Holistic e retorna o frame desenhado"""

        # Converter BGR para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processar frame
        resultados = self.holistic.process(frame_rgb)

        info = {
            'tem_rosto': False,
            'tem_mao_esquerda': False,
            'tem_mao_direita': False,
            'tem_pose': False,
            'num_dedos_esquerda': 0,
            'num_dedos_direita': 0,
            'postura': ''
        }

        # USAR getattr PARA ACESSAR OS ATRIBUTOS DE FORMA SEGURA
        pose_landmarks = getattr(resultados, 'pose_landmarks', None)
        face_landmarks = getattr(resultados, 'face_landmarks', None)
        left_hand_landmarks = getattr(resultados, 'left_hand_landmarks', None)
        right_hand_landmarks = getattr(resultados, 'right_hand_landmarks', None)

        # 1. DESENHAR POSE (CORPO)
        if pose_landmarks:
            info['tem_pose'] = True
            self.mp_desenho.draw_landmarks(
                frame,
                pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.drawing_spec_pose,
                connection_drawing_spec=self.drawing_spec_pose
            )

            # Analisar postura
            info['postura'] = self.analisar_postura(pose_landmarks.landmark)

        # 2. DESENHAR ROSTO
        if face_landmarks:
            info['tem_rosto'] = True
            self.mp_desenho.draw_landmarks(
                frame,
                face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=self.drawing_spec_face,
                connection_drawing_spec=self.drawing_spec_face
            )

        # 3. DESENHAR MÃO ESQUERDA
        if left_hand_landmarks:
            info['tem_mao_esquerda'] = True
            self.mp_desenho.draw_landmarks(
                frame,
                left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.drawing_spec_hands,
                connection_drawing_spec=self.drawing_spec_hands
            )

            # Contar dedos da mão esquerda
            info['num_dedos_esquerda'] = self.contar_dedos(
                left_hand_landmarks.landmark, 'Left')

        # 4. DESENHAR MÃO DIREITA
        if right_hand_landmarks:
            info['tem_mao_direita'] = True
            self.mp_desenho.draw_landmarks(
                frame,
                right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.drawing_spec_hands,
                connection_drawing_spec=self.drawing_spec_hands
            )

            # Contar dedos da mão direita
            info['num_dedos_direita'] = self.contar_dedos(
                right_hand_landmarks.landmark, 'Right')

        return frame, info

    @staticmethod
    def contar_dedos(landmarks, lado):
        """Conta dedos de uma mão"""
        dedos = []

        # Polegar
        if lado == 'Right':
            dedos.append(1 if landmarks[4].x < landmarks[3].x else 0)
        else:
            dedos.append(1 if landmarks[4].x > landmarks[3].x else 0)

        # Outros dedos
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            dedos.append(1 if landmarks[tip].y < landmarks[pip].y else 0)

        return sum(dedos)

    @staticmethod
    def analisar_postura(landmarks):
        """Analisa a postura do corpo"""
        try:
            ombro_esq = landmarks[11]
            ombro_dir = landmarks[12]
            quadril_esq = landmarks[23]
            quadril_dir = landmarks[24]

            # Verificar se está em pé
            if (ombro_esq.y < quadril_esq.y and
                    ombro_dir.y < quadril_dir.y):
                return "Em pé"
            else:
                return "Sentado/Agachado"
        except Exception as erro:
            print(erro)
            return "Indeterminado"


class AppHolistic:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector Holistic - Corpo, Mãos e Rosto")
        self.root.geometry("1000x800")

        # Detector
        self.detector = DetectorHolistic()

        # Controle da câmera
        self.camera = None
        self.is_running = False

        # Configurações de visualização
        self.mostrar_info = True

        # Elementos da interface
        self.video_frame = None
        self.video_label = None
        self.status_var = None
        self.stats_text = None
        self.stats_frame = None

        # Configurar grid weights para expandir
        self.root.grid_rowconfigure(1, weight=1)  # Linha do vídeo expande
        self.root.grid_columnconfigure(0, weight=1)  # Coluna única expande

        self.setup_ui()
        self.iniciar_camera()

    def setup_ui(self):
        """Configura a interface gráfica usando grid"""

        # ===== LINHA 0: Título =====
        titulo = ttk.Label(self.root,
                           text="🦾 DETECTOR HOLISTIC - CORPO, MÃOS E ROSTO 🦾",
                           font=('Arial', 16, 'bold'))
        titulo.grid(row=0, column=0, pady=10, padx=10, sticky="n")

        # ===== LINHA 1: Info cores =====
        info_cores = ttk.Frame(self.root)
        info_cores.grid(row=1, column=0, pady=5, sticky="n")

        ttk.Label(info_cores, text="🟢 Corpo", foreground="green").grid(row=0, column=0, padx=10)
        ttk.Label(info_cores, text="🟣 Rosto", foreground="purple").grid(row=0, column=1, padx=10)
        ttk.Label(info_cores, text="🟡 Mãos", foreground="orange").grid(row=0, column=2, padx=10)

        # ===== LINHA 2: Frame do vídeo (expande) =====
        self.video_frame = ttk.Frame(self.root, relief=tk.SUNKEN, borderwidth=2)
        self.video_frame.grid(row=2, column=0, pady=10, padx=10, sticky="nsew")

        # Configurar grid do video_frame
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        self.video_label = ttk.Label(self.video_frame)
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # ===== LINHA 3: Frame de estatísticas =====
        self.stats_frame = ttk.LabelFrame(self.root, text="📊 Estatísticas", padding=10)
        self.stats_frame.grid(row=3, column=0, pady=10, padx=10, sticky="ew")

        self.stats_frame.grid_columnconfigure(0, weight=1)

        self.stats_text = tk.Text(self.stats_frame, height=8, width=60, font=('Courier', 10))
        self.stats_text.grid(row=0, column=0, sticky="ew")

        # ===== LINHA 4: Botões =====
        button_frame = ttk.Frame(self.root)
        button_frame.grid(row=4, column=0, pady=10, sticky="n")

        ttk.Button(button_frame, text="📸 Tirar Foto",
                   command=self.tirar_foto).grid(row=0, column=0, padx=5)

        ttk.Button(button_frame, text="ℹ️ Info On/Off",
                   command=self.toggle_info).grid(row=0, column=1, padx=5)

        ttk.Button(button_frame, text="❌ Sair",
                   command=self.sair).grid(row=0, column=2, padx=5)

        # ===== LINHA 5: Status bar =====
        self.status_var = tk.StringVar()
        self.status_var.set("Pronto | Holistic ativo")
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=5, column=0, pady=5, padx=10, sticky="ew")

    def iniciar_camera(self):
        """Inicia a captura de vídeo"""
        try:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
            self.camera.set(cv2.CAP_PROP_FPS, 30)

            if not self.camera.isOpened():
                messagebox.showerror("Erro", "Não foi possível abrir a câmera")
                return False

            self.is_running = True
            self.atualizar_frame()
            return True

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao iniciar câmera: {str(e)}")
            return False

    def atualizar_frame(self):
        """Atualiza o frame na interface"""
        if self.is_running and self.camera:
            ret, frame = self.camera.read()

            if ret:
                # Processar frame com Holistic
                frame_processado, info = self.detector.processar_frame(frame)

                if self.mostrar_info:
                    # Adicionar informações no frame
                    y_pos = 30
                    for chave, valor in info.items():
                        if isinstance(valor, bool):
                            texto = f"{chave}: {'✅' if valor else '❌'}"
                        else:
                            texto = f"{chave}: {valor}"

                        cv2.putText(frame_processado, texto,
                                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 255, 255), 1)
                        y_pos += 25

                    # Total de dedos
                    total_dedos = info['num_dedos_esquerda'] + info['num_dedos_direita']
                    cv2.putText(frame_processado, f"TOTAL DEDOS: {total_dedos}",
                                (10, y_pos + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 255), 2)

                # Atualizar estatísticas
                self.atualizar_stats(info)

                # Converter para Tkinter
                frame_rgb = cv2.cvtColor(frame_processado, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)

                # Redimensionar mantendo proporção
                largura_max = 900
                altura_max = 675
                img.thumbnail((largura_max, altura_max), Image.Resampling.LANCZOS)

                imgtk = ImageTk.PhotoImage(image=img)

                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            self.root.after(10, self.atualizar_frame)

    def atualizar_stats(self, info):
        """Atualiza o texto das estatísticas"""
        self.stats_text.delete(1.0, tk.END)

        stats = f"""
🦾 HOLISTIC DETECTOR
━━━━━━━━━━━━━━━━━━━━━━━
👤 Postura: {info['postura']}
👁️ Rosto detectado: {'✅' if info['tem_rosto'] else '❌'}

🤚 Mão Esquerda: {'✅' if info['tem_mao_esquerda'] else '❌'}
   Dedos: {info['num_dedos_esquerda']}

🖐️ Mão Direita: {'✅' if info['tem_mao_direita'] else '❌'}
   Dedos: {info['num_dedos_direita']}

🎯 TOTAL DE DEDOS: {info['num_dedos_esquerda'] + info['num_dedos_direita']}
━━━━━━━━━━━━━━━━━━━━━━━
"""
        self.stats_text.insert(1.0, stats)

        # Atualizar status
        total_dedos = info['num_dedos_esquerda'] + info['num_dedos_direita']
        self.status_var.set(f"Ativo | Dedos: {total_dedos} | Postura: {info['postura']}")

    def tirar_foto(self):
        """Tira uma foto com a detecção atual"""
        if self.camera and self.is_running:
            ret, frame = self.camera.read()
            if ret:
                # Criar pasta
                if not os.path.exists("fotos_holistic"):
                    os.makedirs("fotos_holistic")

                # Processar frame
                frame_processado, info = self.detector.processar_frame(frame)

                # Salvar
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nome = f"fotos_holistic/holistic_{timestamp}.jpg"
                cv2.imwrite(nome, frame_processado)

                self.status_var.set(f"📸 Foto salva: {nome}")
                messagebox.showinfo("Sucesso", f"Foto salva como:\n{nome}")

    def toggle_info(self):
        """Alterna a exibição das informações na tela"""
        self.mostrar_info = not self.mostrar_info

    def sair(self):
        """Fecha o aplicativo"""
        self.is_running = False
        if self.camera:
            self.camera.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = AppHolistic(root)
    root.protocol("WM_DELETE_WINDOW", app.sair)
    root.mainloop()


if __name__ == "__main__":
    main()