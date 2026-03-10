import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
import os
from datetime import datetime


class DetectorMaos:
    def __init__(self):
        self.mp_maos = mp.solutions.hands
        self.maos = self.mp_maos.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_desenho = mp.solutions.drawing_utils

    @staticmethod
    def contar_dedos(landmarks, lado_mao):
        """Conta quantos dedos estão levantados"""
        dedos = []

        # Polegar
        if lado_mao == 'Right':
            dedos.append(1 if landmarks[4].x < landmarks[3].x else 0)
        else:
            dedos.append(1 if landmarks[4].x > landmarks[3].x else 0)

        # Outros dedos
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            dedos.append(1 if landmarks[tip].y < landmarks[pip].y else 0)

        return sum(dedos)


class AppDetectorMaos:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Mãos - Contador de Dedos")
        self.root.geometry("900x700")

        # ===== TODOS OS ATRIBUTOS DE INSTÂNCIA AQUI =====

        # Detector
        self.detector = DetectorMaos()

        # Controle da câmara
        self.camera = None
        self.is_running = False

        # Configurações de visualização
        self.modo_cores = False
        self.detectar_maos = True

        # Elementos da interface (inicializados como None)
        self.video_frame = None
        self.video_label = None
        self.label_maos = None
        self.label_dedos = None
        self.btn_tirar_foto = None
        self.btn_modo_cores = None
        self.btn_toggle_deteccao = None
        self.btn_sair = None
        self.status_var = None

        # Configurar interface (agora que os atributos existem)
        self.setup_ui()

        # Iniciar câmera
        self.iniciar_camera()

    def setup_ui(self):
        """Configura a interface gráfica"""

        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Título
        titulo = ttk.Label(main_frame,
                           text="🖐️ DETECTOR DE MÃOS - CONTADOR DE DEDOS 🖐️",
                           font=('Arial', 16, 'bold'))
        titulo.pack(pady=10)

        # Frame do vídeo (atributo definido no __init__)
        self.video_frame = ttk.Frame(main_frame, relief=tk.SUNKEN, borderwidth=2)
        self.video_frame.pack(padx=10, pady=5)

        # Label do vídeo (atributo definido no __init__)
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack()

        # Frame de informações
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(pady=10, fill=tk.X)

        # Labels para mostrar contagem (atributos definidos no __init__)
        self.label_maos = ttk.Label(info_frame,
                                    text="Mãos detectadas: 0",
                                    font=('Arial', 12))
        self.label_maos.pack()

        self.label_dedos = ttk.Label(info_frame,
                                     text="Total de dedos: 0",
                                     font=('Arial', 14, 'bold'))
        self.label_dedos.pack()

        # Frame dos botões
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)

        # Botões de controle (atributos definidos no __init__)
        self.btn_tirar_foto = ttk.Button(button_frame,
                                         text="📸 Tirar Foto",
                                         command=self.tirar_foto)
        self.btn_tirar_foto.pack(side=tk.LEFT, padx=5)

        self.btn_modo_cores = ttk.Button(button_frame,
                                         text="🌈 Modo Cores",
                                         command=self.toggle_modo_cores)
        self.btn_modo_cores.pack(side=tk.LEFT, padx=5)

        self.btn_toggle_deteccao = ttk.Button(button_frame,
                                              text="⏸️ Pausar Detecção",
                                              command=self.toggle_deteccao)
        self.btn_toggle_deteccao.pack(side=tk.LEFT, padx=5)

        self.btn_sair = ttk.Button(button_frame,
                                   text="❌ Sair",
                                   command=self.sair)
        self.btn_sair.pack(side=tk.LEFT, padx=5)

        # Status bar (atributo definido no __init__)
        self.status_var = tk.StringVar()
        self.status_var.set("Pronto | Modo: Detecção de mãos ativa")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=5)

    def iniciar_camera(self):
        """Inicia a captura de vídeo"""
        try:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if not self.camera.isOpened():
                messagebox.showerror("Erro", "Não foi possível abrir a câmera")
                return False

            self.is_running = True
            self.atualizar_frame()
            return True

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao iniciar câmera: {str(e)}")
            return False

    def processar_maos(self, frame):
        """Processa o frame para detectar mãos e contar dedos"""

        # Converter BGR para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processar frame
        resultados = self.detector.maos.process(frame_rgb)

        total_dedos = 0
        num_maos = 0

        # Acessar atributos de forma segura
        multi_hand_landmarks = getattr(resultados, 'multi_hand_landmarks', None)
        multi_handedness = getattr(resultados, 'multi_handedness', None)

        if multi_hand_landmarks:
            num_maos = len(multi_hand_landmarks)

            for i, landmarks in enumerate(multi_hand_landmarks):
                # Desenhar landmarks
                if self.modo_cores:
                    cor = (0, 255, 0) if i == 0 else (255, 0, 0)
                    self.detector.mp_desenho.draw_landmarks(
                        frame, landmarks, self.detector.mp_maos.HAND_CONNECTIONS,
                        self.detector.mp_desenho.DrawingSpec(color=cor, thickness=2),
                        self.detector.mp_desenho.DrawingSpec(color=cor, thickness=1)
                    )
                else:
                    self.detector.mp_desenho.draw_landmarks(
                        frame, landmarks, self.detector.mp_maos.HAND_CONNECTIONS
                    )

                # Obter lado da mão
                lado_mao = "Unknown"
                if multi_handedness and i < len(multi_handedness):
                    classificacao = multi_handedness[i]
                    classification_list = getattr(classificacao, 'classification', None)
                    if classification_list and len(classification_list) > 0:
                        lado_mao = getattr(classification_list[0], 'label', 'Unknown')

                # Contar dedos
                dedos_levantados = DetectorMaos.contar_dedos(landmarks.landmark, lado_mao)
                total_dedos += dedos_levantados

                # Desenhar contador
                h, w, _ = frame.shape
                x = int(landmarks.landmark[0].x * w)
                y = int(landmarks.landmark[0].y * h) - 20

                cv2.rectangle(frame, (x - 10, y - 30), (x + 70, y), (0, 0, 0), -1)
                cv2.putText(frame, f"{lado_mao}: {dedos_levantados}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)

        return frame, num_maos, total_dedos

    def atualizar_frame(self):
        """Atualiza o frame na interface"""
        if self.is_running and self.camera:
            ret, frame = self.camera.read()

            if ret:
                frame_processado = frame.copy()

                if self.detectar_maos:
                    frame_processado, num_maos, total_dedos = self.processar_maos(frame_processado)

                    # Atualizar interface (atributos definidos no __init__)
                    if self.label_maos:
                        self.label_maos.config(text=f"Mãos detectadas: {num_maos}")
                    if self.label_dedos:
                        self.label_dedos.config(text=f"Total de dedos: {total_dedos}")

                    cv2.putText(frame_processado, f"Dedos: {total_dedos}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2)

                # Converter para Tkinter
                frame_rgb = cv2.cvtColor(frame_processado, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((800, 600), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)

                if self.video_label:
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)

            self.root.after(10, self.atualizar_frame)

    def tirar_foto(self):
        """Tira uma foto com a detecção atual"""
        if self.camera and self.is_running:
            ret, frame = self.camera.read()
            if ret:
                # Criar pasta
                if not os.path.exists("fotos_maos"):
                    os.makedirs("fotos_maos")

                # Processar frame
                frame_processado, num_maos, total_dedos = self.processar_maos(frame.copy())

                # Adicionar info na foto
                cv2.putText(frame_processado, f"Maos: {num_maos} | Dedos: {total_dedos}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

                # Salvar
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nome = f"fotos_maos/maos_{timestamp}.jpg"
                cv2.imwrite(nome, frame_processado)

                if self.status_var:
                    self.status_var.set(f"📸 Foto salva: {nome}")
                messagebox.showinfo("Sucesso", f"Foto salva como:\n{nome}")

    def toggle_modo_cores(self):
        """Alterna entre modo normal e colorido"""
        self.modo_cores = not self.modo_cores
        if self.modo_cores:
            if self.btn_modo_cores:
                self.btn_modo_cores.config(text="⚫ Modo Normal")
            if self.status_var:
                self.status_var.set("Modo colorido ativado")
        else:
            if self.btn_modo_cores:
                self.btn_modo_cores.config(text="🌈 Modo Cores")
            if self.status_var:
                self.status_var.set("Modo normal ativado")

    def toggle_deteccao(self):
        """Pausa/retoma a detecção"""
        self.detectar_maos = not self.detectar_maos
        if self.detectar_maos:
            if self.btn_toggle_deteccao:
                self.btn_toggle_deteccao.config(text="⏸️ Pausar Detecção")
            if self.status_var:
                self.status_var.set("Detecção ativada")
        else:
            if self.btn_toggle_deteccao:
                self.btn_toggle_deteccao.config(text="▶️ Retomar Detecção")
            if self.status_var:
                self.status_var.set("Detecção pausada")

    def sair(self):
        """Fecha o aplicativo"""
        self.is_running = False
        if self.camera:
            self.camera.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = AppDetectorMaos(root)
    root.protocol("WM_DELETE_WINDOW", app.sair)
    root.mainloop()


if __name__ == "__main__":
    main()
