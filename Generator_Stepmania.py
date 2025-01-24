<<<<<<< HEAD
"""
Generator_Stepmania.py
Este script genera archivos para el juego StepMania basándose en el análisis de archivos de audio.
Analiza la música para crear patrones de pasos que coincidan con el ritmo y permite incluir videos de fondo.
"""

# Importaciones básicas del sistema y utilidades
import os
import random
import shutil
import traceback
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

# Intentamos importar las bibliotecas opcionales con manejo de errores elegante
try:
    import cv2
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("Advertencia: moviepy no está disponible. El procesamiento de video será limitado.")

try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Error: librosa es necesario para el análisis musical.")
    raise

@dataclass
class DanceStep:
    """
    Representa un paso de baile con sus características específicas.
    
    Attributes:
        direction: Dirección del paso ('left', 'right', 'up', 'down', 'none', o versiones 'hold_')
        timing: Momento exacto del paso en segundos
        duration: Duración del paso en segundos
        beat_strength: Intensidad del beat musical asociado (0.0 a 1.0)
    """
    direction: str
    timing: float
    duration: float
    beat_strength: float

class DDRSongGenerator:
    """
    Clase principal para generar archivos de canciones para StepMania/DDR.
    Esta clase maneja todo el proceso de análisis musical y generación de pasos.
    """
    
    def __init__(self, audio_path: str, video_path: Optional[str] = None, stepmania_path: Optional[str] = None):
        """
        Inicializa el generador con los archivos necesarios y analiza la música.
        
        Args:
            audio_path: Ruta al archivo de audio principal
            video_path: Ruta opcional al archivo de video de fondo
            stepmania_path: Ruta opcional a la instalación de StepMania
        """
        # Verificamos que el archivo de audio existe
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"No se encontró el archivo de audio: {audio_path}")

        # Verificación inicial de video
        if video_path and not MOVIEPY_AVAILABLE:
            print("Advertencia: Se proporcionó video pero moviepy no está disponible.")
            print("El video no será procesado. Instale moviepy para habilitar esta función.")
            video_path = None
        elif video_path and not os.path.exists(video_path):
            print(f"Advertencia: No se encontró el archivo de video: {video_path}")
            video_path = None

        # Configuración de rutas
        self.audio_path = os.path.abspath(audio_path)
        self.video_path = os.path.abspath(video_path) if video_path else None
        self.stepmania_path = stepmania_path or self._find_stepmania_path()

        # Análisis musical inicial
        print("Analizando audio...")
        try:
            self.y, self.sr = librosa.load(audio_path)
            self.tempo, self.beat_frames = librosa.beat.beat_track(y=self.y, sr=self.sr)
            self.beat_times = librosa.frames_to_time(self.beat_frames, sr=self.sr)
            print(f"Tempo detectado: {self.tempo:.1f} BPM")
        except Exception as e:
            print(f"Error en el análisis de audio: {str(e)}")
            print("Usando configuración predeterminada...")
            self.sr = 44100
            self.tempo = 120
            duration = os.path.getsize(audio_path) / (self.sr * 2)
            self.beat_times = np.arange(0, duration, 60.0 / self.tempo)

        # Análisis avanzado
        self.stops = self._analyze_music_stops()
        self._initialize_difficulty_settings()
        
        # Lista para almacenar los pasos generados
        self.steps: List[DanceStep] = []

    def _find_stepmania_path(self) -> Optional[str]:
        """
        Busca la instalación de StepMania en ubicaciones comunes del sistema.
        Si no encuentra una instalación, crea una carpeta temporal para desarrollo.
        """
        possible_paths = [
            "C:/Program Files/Stepmania 5",
            "C:/Program Files (x86)/Stepmania 5",
            "D:/StepMania",
            "D:/Games/Stepmania",
            "E:/Games/Stepmania",
            "./Stepmania",  # Ruta relativa para desarrollo
            "../Stepmania"  # Ruta relativa para desarrollo
        ]
        
        # Buscamos en las rutas predefinidas
        for path in possible_paths:
            if os.path.exists(path):
                print(f"StepMania encontrado en: {path}")
                return path

        # Si no encontramos StepMania, creamos una carpeta temporal
        temp_path = "./StepmaniaTemp"
        os.makedirs(temp_path, exist_ok=True)
        print(f"No se encontró StepMania. Usando carpeta temporal: {temp_path}")
        return temp_path

    def _initialize_difficulty_settings(self):
        """
        Configura los parámetros específicos para cada nivel de dificultad.
        Define patrones de pasos, probabilidades y configuraciones para cada nivel.
        """
        self.difficulty_settings = {
            'beginner': {
                'level': 2,
                'ratings': '0.135,0.213,0.036,0.081,0.000',
                'max_steps_per_beat': 1,
                'holds_probability': 0.1,
                'empty_measures_intro': 2,
                'patterns': [['left'], ['right'], ['up'], ['down']]
            },
            'easy': {
                'level': 4,
                'ratings': '0.263,0.319,0.108,0.009,0.000',
                'max_steps_per_beat': 1,
                'holds_probability': 0.15,
                'empty_measures_intro': 1,
                'patterns': [['left', 'right'], ['up', 'down']]
            },
            'normal': {
                'level': 6,
                'ratings': '0.514,0.532,0.189,0.379,0.000',
                'max_steps_per_beat': 2,
                'holds_probability': 0.2,
                'empty_measures_intro': 1,
                'patterns': [['left', 'right', 'up'], ['down', 'left', 'right']]
            },
            'hard': {
                'level': 9,
                'ratings': '0.740,0.710,0.523,0.216,0.054',
                'max_steps_per_beat': 3,
                'holds_probability': 0.25,
                'empty_measures_intro': 0,
                'patterns': [['left', 'right', 'up', 'down'], ['up', 'down', 'left', 'right']]
            }
        }


    def _analyze_music_stops(self) -> List[Dict[str, float]]:
        """
        Detecta momentos de pausa significativa en la música usando un análisis más selectivo.
        Solo detecta pausas en cambios muy dramáticos de intensidad y usa duraciones más cortas.
        """
        stops = []
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        
        # Calculamos un umbral más exigente
        mean_strength = np.mean(onset_env)
        std_strength = np.std(onset_env)
        threshold = mean_strength - (std_strength * 2)  # Más selectivo
        
        # Buscamos segmentos con caídas significativas de intensidad
        window_size = 4  # Ventana para analizar el contexto
        for i in range(window_size, len(onset_env) - window_size):
            # Verificamos si hay una caída significativa
            before = np.mean(onset_env[i-window_size:i])
            current = onset_env[i]
            after = np.mean(onset_env[i+1:i+1+window_size])
            
            if (before > threshold and 
                current < threshold and 
                after > threshold):
                
                time = librosa.frames_to_time(i, sr=self.sr)
                
                # Calculamos una duración más apropiada basada en el BPM
                beat_duration = 60.0 / self.tempo
                stop_duration = beat_duration * 0.25  # 1/4 de beat
                
                stops.append({
                    'time': time,
                    'duration': min(0.200, stop_duration)  # Máximo 200ms
                })
        
        # Filtramos pausas muy cercanas
        filtered_stops = []
        last_time = -1
        min_interval = 60.0 / self.tempo  # Un beat de separación
        
        for stop in stops:
            if last_time == -1 or (stop['time'] - last_time) >= min_interval:
                filtered_stops.append(stop)
                last_time = stop['time']
        
        return filtered_stops
    
    # def _analyze_music_stops(self) -> List[Dict[str, float]]:
    #     """
    #     Detecta momentos de pausa significativa en la música usando análisis de onsets.
    #     Estas paradas se utilizarán para generar momentos de pausa en los pasos.
    #     """
    #     stops = []
    #     onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        
    #     # Buscamos puntos donde la intensidad cae significativamente
    #     threshold = np.mean(onset_env) * 0.3
    #     for i in range(1, len(onset_env)):
    #         if onset_env[i] < threshold and onset_env[i-1] > threshold:
    #             time = librosa.frames_to_time(i, sr=self.sr)
    #             stops.append({
    #                 'time': time,
    #                 'duration': 0.620  # Duración estándar de pausa
    #             })
        
    #     return stops

    def _analyze_beat_strength(self) -> List[Dict[str, float]]:
        """
        Realiza un análisis detallado de las características musicales de cada beat.
        Esto permite sincronizar los pasos con diferentes elementos de la música.
        """
        print("Analizando intensidad de beats...")
        beat_info = []
        hop_length = 512
        
        # Análisis del espectrograma mel para diferentes bandas de frecuencia
        spec = librosa.feature.melspectrogram(
            y=self.y, 
            sr=self.sr,
            n_mels=128,
            hop_length=hop_length
        )
        
        # Detección de onsets para ritmo
        onset_env = librosa.onset.onset_strength(
            y=self.y, 
            sr=self.sr,
            hop_length=hop_length,
            aggregate=np.median
        )
        
        # Análisis de contraste espectral para cambios dramáticos
        spectral_contrast = librosa.feature.spectral_contrast(
            y=self.y, 
            sr=self.sr,
            hop_length=hop_length
        )
        
        # Analizamos cada beat individual
        for beat_time in self.beat_times:
            frame = librosa.time_to_frames(beat_time, sr=self.sr, hop_length=hop_length)
            if frame < len(onset_env):
                strength = onset_env[frame]
                
                if frame < spec.shape[1]:
                    spec_beat = spec[:, frame]
                    # Analizamos diferentes rangos de frecuencia
                    bass_strength = np.mean(spec_beat[:20])    # Bajos
                    mid_strength = np.mean(spec_beat[20:80])   # Medios
                    high_strength = np.mean(spec_beat[80:])    # Altos
                    contrast = np.mean(spectral_contrast[:, frame])
                else:
                    bass_strength = mid_strength = high_strength = contrast = 0.0
                
                beat_info.append({
                    'time': float(beat_time),
                    'strength': float(strength),
                    'bass_strength': float(bass_strength),
                    'mid_strength': float(mid_strength),
                    'high_strength': float(high_strength),
                    'contrast': float(contrast)
                })
        
        return beat_info
    
    def generate_sequence(self, difficulty: str = 'normal') -> List[DanceStep]:
        """
        Genera la secuencia completa de pasos para toda la duración de la canción.
        
        Args:
            difficulty: Nivel de dificultad deseado ('beginner', 'easy', 'normal', 'hard')
        Returns:
            Lista de pasos de baile sincronizados con la música
        """
        settings = self.difficulty_settings[difficulty.lower()]
        beat_info = self._analyze_beat_strength()
        self.steps = []
        
        # 1. Calculamos la duración exacta y los beats totales
        song_duration = librosa.get_duration(y=self.y, sr=self.sr)
        
        # Manejamos el tempo de manera segura
        try:
            if hasattr(self.tempo, 'item'):
                tempo = float(self.tempo.item())
            else:
                tempo = float(self.tempo)
        except (AttributeError, IndexError, TypeError):
            print("Advertencia: Usando tempo predeterminado")
            tempo = 120.0

        # Calculamos los beats totales usando la duración real de la canción
        beats_per_second = tempo / 60
        total_beats = int(np.ceil(song_duration * beats_per_second * 4))  # Multiplicamos por 4 para las subdivisiones
        beats_per_measure = 16  # Cada compás tiene 4 beats × 4 subdivisiones
        
        # Calculamos el número total de compases necesarios
        min_measures = int(np.ceil(song_duration * tempo / 240))  # 240 = 60 segundos * 4 beats por compás
        total_measures = max(
            int(np.ceil(total_beats / beats_per_measure)),
            min_measures,
            len(self.beat_times)  # Aseguramos que cubrimos todos los beats detectados
        )
        
        # Mostramos información de diagnóstico
        print(f"Duración de la canción: {song_duration:.2f} segundos")
        print(f"Tempo: {tempo:.2f} BPM")
        print(f"Beats detectados: {len(self.beat_times)}")
        print(f"Total de beats a generar: {total_beats}")
        print(f"Total de compases a generar: {total_measures}")
        
        # Generamos los compases vacíos iniciales (introducción)
        empty_measures = settings['empty_measures_intro']
        for measure in range(empty_measures):
            for _ in range(beats_per_measure):
                self.steps.append(DanceStep(
                    direction='none',
                    timing=len(self.steps),
                    duration=0.5,
                    beat_strength=0
                ))
        
        # Generamos el resto de los compases
        for measure in range(empty_measures, total_measures):
            measure_start_beat = measure * beats_per_measure
            steps_in_measure = 0
            
            while steps_in_measure < beats_per_measure:
                current_beat = measure_start_beat + steps_in_measure
                
                # Usamos módulo para el índice del beat para repetir el patrón si es necesario
                beat_idx = current_beat % len(beat_info) if beat_info else 0
                beat = beat_info[beat_idx] if beat_info else {'strength': 0.5, 'bass_strength': 0.5}
                
                # Decidimos si poner un paso o espacio vacío
                if (beat['strength'] > 0.3 or 
                    beat['bass_strength'] > 0.3 or 
                    steps_in_measure % 8 == 0):  # Aseguramos al menos un paso cada 8 beats
                    
                    # Seleccionamos la dirección del paso
                    direction = random.choice(['left', 'down', 'up', 'right'])
                    
                    # Añadimos el paso
                    self.steps.append(DanceStep(
                        direction=direction,
                        timing=current_beat,
                        duration=0.5,
                        beat_strength=beat['strength']
                    ))
                    steps_in_measure += 1
                    
                    # Añadimos tres espacios vacíos después del paso
                    for _ in range(min(3, beats_per_measure - steps_in_measure)):
                        self.steps.append(DanceStep(
                            direction='none',
                            timing=current_beat + 1,
                            duration=0.5,
                            beat_strength=0
                        ))
                        steps_in_measure += 1
                else:
                    # Añadimos un espacio vacío
                    self.steps.append(DanceStep(
                        direction='none',
                        timing=current_beat,
                        duration=0.5,
                        beat_strength=0
                    ))
                    steps_in_measure += 1
        
        # Verificación final con cálculos más precisos
        real_duration = len(self.steps) / (tempo / 15)  # 15 = 60/4 para considerar subdivisiones
        print(f"Pasos generados: {len(self.steps)}")
        print(f"Duración esperada: {song_duration:.2f} segundos")
        print(f"Duración real: {real_duration:.2f} segundos")
        
        return self.steps
    
    
    
    # def generate_sequence(self, difficulty: str = 'normal') -> List[DanceStep]:
    #     """
    #     Genera la secuencia de pasos basada en el análisis musical y el nivel de dificultad.
    #     """
    #     settings = self.difficulty_settings[difficulty.lower()]
    #     beat_info = self._analyze_beat_strength()
    #     self.steps = []
        
    #     # Calculamos parámetros básicos de manera segura
    #     song_duration = librosa.get_duration(y=self.y, sr=self.sr)
        
    #     try:
    #         if hasattr(self.tempo, 'item'):
    #             tempo = float(self.tempo.item())
    #         else:
    #             tempo = float(self.tempo)
    #     except (AttributeError, IndexError, TypeError):
    #         print("Advertencia: Usando tempo predeterminado")
    #         tempo = 120.0

    #     beats_per_second = tempo / 60
    #     total_beats = int(song_duration * beats_per_second)
        
    #     if difficulty.lower() == 'beginner':
    #         empty_measures = settings['empty_measures_intro']
    #         current_beat = empty_measures * 4
            
    #         # Generamos los compases vacíos iniciales
    #         for _ in range(current_beat):
    #             self.steps.append(DanceStep(
    #                 direction='none',
    #                 timing=current_beat,
    #                 duration=0.5,
    #                 beat_strength=0
    #             ))
    #             current_beat += 1
        
    #         # Generamos los pasos principales con un ritmo más dinámico
    #         while current_beat < total_beats:
    #             beat_idx = min(int(current_beat * len(beat_info) / total_beats), len(beat_info) - 1)
    #             beat = beat_info[beat_idx]
            
    #             # Reducimos umbral y mejoramos la lógica de generación de pasos
    #             if (beat['strength'] > 0.3 or beat['bass_strength'] > 0.3 or 
    #                 current_beat % 8 == 0):  # Asegura pasos regulares
                    
    #                 direction = random.choice(['left', 'down', 'up', 'right'])
                    
    #                 # Un paso seguido de tres beats vacíos (patrón 1-3)
    #                 self.steps.append(DanceStep(
    #                     direction=direction,
    #                     timing=current_beat,
    #                     duration=0.5,
    #                     beat_strength=beat['strength']
    #                 ))
    #                 current_beat += 1
                    
    #                 # Tres beats vacíos
    #                 for _ in range(3):
    #                     self.steps.append(DanceStep(
    #                         direction='none',
    #                         timing=current_beat,
    #                         duration=0.5,
    #                         beat_strength=0
    #                     ))
    #                     current_beat += 1
    #             else:
    #                 # Beat débil, un solo paso vacío
    #                 self.steps.append(DanceStep(
    #                     direction='none',
    #                     timing=current_beat,
    #                     duration=0.5,
    #                     beat_strength=0
    #                 ))
    #             current_beat += 1
    # def generate_sequence(self, difficulty: str = 'normal') -> List[DanceStep]:
    #     """
    #     Genera la secuencia de pasos basada en el análisis musical y el nivel de dificultad.
    #     """
    #     settings = self.difficulty_settings[difficulty.lower()]
    #     beat_info = self._analyze_beat_strength()
    #     self.steps = []
        
    #     # Calculamos parámetros básicos
    #     song_duration = librosa.get_duration(y=self.y, sr=self.sr)
    #     beats_per_second = float(self.tempo[0] if hasattr(self.tempo, 'item') else self.tempo) / 60
    #     total_beats = int(song_duration * beats_per_second)

    #     if difficulty.lower() == 'beginner':
    #         empty_measures = 2
    #         current_beat = empty_measures * 4

    #     # Secure time handling
    #     try:
    #         if hasattr(self.tempo, 'item'):
    #             tempo = float(self.tempo.item())
    #         else:
    #             tempo = float(self.tempo)
    #     except (AttributeError, IndexError, TypeError):
    #         print("Advertencia: Usando tempo predeterminado")
    #         tempo = 120.0

    #     beats_per_second = tempo / 60
    #     total_beats = int(song_duration * beats_per_second)
        
    #     # Añadimos compases vacíos al inicio para dar tiempo al jugador
    #     empty_measures = settings['empty_measures_intro']
    #     current_beat = empty_measures * 4
        
    #     for _ in range(current_beat):
    #         self.steps.append(DanceStep(
    #             direction='none',
    #             timing=current_beat,
    #             duration=0.5,
    #             beat_strength=0
    #         ))
    #         current_beat += 1
        
    #     # Generamos los pasos principales
    #     while current_beat < total_beats:
    #         beat_idx = min(int(current_beat * len(beat_info) / total_beats), len(beat_info) -1)
    #         beat = beat_info[beat_idx]
                        
    #         if beat['strength'] > 0.3 or beat['bass_strength'] > 0.3:
    #             direction = random.choice(['left', 'down', 'up', 'right'])
                 
    #             self.steps.append(DanceStep(
    #                 direction=direction,
    #                 timing=current_beat,
    #                 duration=0.5,
    #                 beat_strength=beat['strength']
    #             ))
    #             current_beat += 1

    #             for _ in range(3):
    #                 self.steps.append(DanceStep(
    #                     direction='none',
    #                     timing=current_beat,
    #                     duration=0.5,
    #                     beat_strength=0
    #                 ))
    #                 current_beat += 1

    #         else:
    #             # En beats débiles, añadimos paso vacío
    #             self.steps.append(DanceStep(
    #                 direction='none',
    #                 timing=current_beat,
    #                 duration=0.5,
    #                 beat_strength=0
    #             ))
    #             current_beat += 1
        
    #     return self.steps

    def process_background_video(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Procesa el video de fondo para máxima compatibilidad con StepMania 3.9
        """
        if not self.video_path or not MOVIEPY_AVAILABLE:
            return None, None, None
            
        try:
            video = VideoFileClip(self.video_path)
            
            # Ajustamos tamaño para StepMania 3.9
            target_size = (640, 480)
            processed_video = video.resize(target_size)
            
            # Extraemos frames para banner y background
            banner_time = video.duration * 0.25
            bg_time = video.duration * 0.75
            
            banner_frame = video.get_frame(banner_time)
            bg_frame = video.get_frame(bg_time)
                
            # Procesamos los frames
            banner = cv2.resize(banner_frame, (256, 80))
            background = cv2.resize(bg_frame, target_size)
                
            # Preparamos rutas de salida con formato más compatible
            base_path = os.path.splitext(self.video_path)[0]
            video_out = f"{base_path}_processed.mpg"  # Cambiamos a MPEG-1
            banner_out = f"{base_path}-bn.png"
            bg_out = f"{base_path}-bg.png"
            
            # Guardamos el video con configuración más compatible
            processed_video.write_videofile(
                video_out,
                codec='mpeg1video',  # Usamos MPEG-1
                audio=False,
                fps=30,
                bitrate="2000k"  # Bajamos el bitrate para mejor compatibilidad
            )
            
            cv2.imwrite(banner_out, cv2.cvtColor(banner, cv2.COLOR_RGB2BGR))
            cv2.imwrite(bg_out, cv2.cvtColor(background, cv2.COLOR_RGB2BGR))
            
            return video_out, banner_out, bg_out
            
        except Exception as e:
            print(f"Error procesando video: {str(e)}")
            return self.video_path, None, None

    # def process_background_video(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    #     """
    #     Procesa el video de fondo para adaptarlo a los requisitos de StepMania.
    #     Genera el video principal, banner y background.
    #     """
    #     if not self.video_path or not MOVIEPY_AVAILABLE:
    #         return None, None, None
            
    #     try:
    #         video = VideoFileClip(self.video_path)
            
    #         Ajustamos el video al tamaño estándar de StepMania
    #         target_size = (640, 480)
    #         processed_video = video.resize(target_size)
            
    #         Extraemos frames para banner y background
    #         banner_time = video.duration * 0.25
    #         bg_time = video.duration * 0.75
            
    #         banner_frame = video.get_frame(banner_time)
    #         bg_frame = video.get_frame(bg_time)
            
    #         Procesamos los frames
    #         banner = cv2.resize(banner_frame, (256, 80))
    #         background = cv2.resize(bg_frame, target_size)
            
    #         Preparamos rutas de salida
    #         base_path = os.path.splitext(self.video_path)[0]
    #         video_out = f"{base_path}_processed.avi"
    #         banner_out = f"{base_path}-bn.png"
    #         bg_out = f"{base_path}-bg.png"
            
    #         Guardamos los archivos
    #         processed_video.write_videofile(
    #             video_out,
    #             codec='libx264',
    #             audio=False,
    #             fps=30
    #         )
            
    #         cv2.imwrite(banner_out, cv2.cvtColor(banner, cv2.COLOR_RGB2BGR))
    #         cv2.imwrite(bg_out, cv2.cvtColor(background, cv2.COLOR_RGB2BGR))
            
    #         return video_out, banner_out, bg_out
            
    #     except Exception as e:
    #         print(f"Error procesando video: {str(e)}")
    #         return self.video_path, None, None

    def _generate_sm_content(self, title: str, difficulties: List[str]) -> str:
        """
        Genera el contenido del archivo .sm con el formato requerido por StepMania.
        """
        try:
        # Si self.tempo es un array de numpy, obtenemos el primer valor
            if hasattr(self.tempo, 'item'):
                tempo = float(self.tempo.item())
        # Si es un número normal, lo convertimos directamente
            else:
                tempo = float(self.tempo)
        except (AttributeError, IndexError, TypeError):
            # Si algo falla, usamos un valor predeterminado seguro
            print("Advertencia: No se pudo determinar el tempo exacto. Usando valor predeterminado.")
            tempo = 120.0
        
        # Procesamos video si existe
        video_path, banner_path, bg_path = self.process_background_video() if self.video_path else (None, None, None)
        
        # Construimos el encabezado
        audio_filename = os.path.basename(self.audio_path)
        content = f"""#TITLE:{title};
#SUBTITLE:;
#ARTIST:Generated;
#TITLETRANSLIT:;
#SUBTITLETRANSLIT:;
#ARTISTTRANSLIT:;
#GENRE:;
#CREDIT:Generated by Python;
#BANNER:{os.path.basename(banner_path) if banner_path else ''};
#BACKGROUND:{os.path.basename(bg_path) if bg_path else ''};
#LYRICSPATH:;
#CDTITLE:;
#MUSIC:audio{os.path.splitext(audio_filename)[1]};
#OFFSET:-0.100;
#SAMPLESTART:52.840;
#SAMPLELENGTH:13.460;
#SELECTABLE:YES;
#DISPLAYBPM:{tempo:.3f};
#BPMS:0.000={tempo:.3f};"""

        # Añadimos información de paradas musicales
        if self.stops:
            stops_str = ','.join([f"{stop['time']:.3f}={stop['duration']:.3f}" for stop in self.stops])
            content += f"\n#STOPS:{stops_str};"
        else:
            content += "\n#STOPS:;"

        # Configuramos video de fondo
        if video_path:
            video_filename = os.path.basename(video_path)
            content += f"\n#BGCHANGES:-0.950={video_filename}=1.000=1=1=0,\n99999=-nosongbg-=1.000=0=0=0 // don't automatically add -songbackground-\n;"
        else:
            content += "\n#BGCHANGES:;"

        # Generamos notas para cada dificultad
        for diff in difficulties:
            settings = self.difficulty_settings[diff.lower()]
            self.generate_sequence(diff)
            content += f"""
//---------------dance-single - ----------------
#NOTES:
     dance-single:
     :
     {diff.capitalize()}:
     {settings['level']}:
     {settings['ratings']}:
{self._convert_steps_to_measures()}
;"""

        return content

    def _convert_steps_to_measures(self) -> str:
        """
        Convierte los pasos al formato de medidas de StepMania.
        """
        if not self.steps:
            return "0000\n0000\n0000\n0000\n"
        
        BEATS_PER_MEASURE = 4
        SUBDIVISIONS = 4
        measure_str = ""
        
        for i in range(0, len(self.steps), BEATS_PER_MEASURE * SUBDIVISIONS):
            measure_steps = self.steps[i:i + BEATS_PER_MEASURE * SUBDIVISIONS]
            measure_content = ""
            
            for step in measure_steps:
                # Convertimos cada paso al formato de flechas
                if step.direction == 'left':
                    arrow = "1000"
                elif step.direction == 'down':
                    arrow = "0100"
                elif step.direction == 'up':
                    arrow = "0010"
                elif step.direction == 'right':
                    arrow = "0001"
                elif step.direction.startswith('hold_'):
                    base_dir = step.direction.split('_')[1]
                    arrow = self._get_hold_arrow(base_dir)
                else:
                    arrow = "0000"
                
                measure_content += arrow + "\n"
            
            # Completamos la medida si es necesario
            while len(measure_content.split('\n')) < BEATS_PER_MEASURE * SUBDIVISIONS + 1:
                measure_content += "0000\n"
            
            measure_str += measure_content + ",\n"
        
        return measure_str

    def _get_hold_arrow(self, direction: str) -> str:
        """
        Convierte una dirección de flecha sostenida al formato de StepMania.
        """
        hold_map = {
            'left': '2000',
            'down': '0200',
            'up': '0020',
            'right': '0002'
        }
        return hold_map.get(direction, '0000')

    def create_stepmania_files(self, song_title: str, difficulty: str = 'normal') -> str:
        """
        Crea todos los archivos necesarios para la canción en StepMania.
        """
        if not self.stepmania_path:
            raise ValueError("No se pudo encontrar la instalación de StepMania")
        
        # Creamos el directorio para la canción
        songs_path = os.path.join(self.stepmania_path, 'Songs')
        song_dir = os.path.join(songs_path, song_title)
        os.makedirs(song_dir, exist_ok=True)
        
        # Generamos archivos
        difficulties = ['beginner', 'easy', 'normal', 'hard'] if difficulty == 'all' else [difficulty]
        sm_content = self._generate_sm_content(song_title, difficulties)
        
        # Copiamos el archivo de audio
        print("Copiando archivos...")
        audio_ext = os.path.splitext(self.audio_path)[1]
        shutil.copy2(self.audio_path, os.path.join(song_dir, f"audio{audio_ext}"))
        
        # Procesamos y copiamos archivos de video
        if self.video_path:
            video_path, banner_path, bg_path = self.process_background_video()
            if video_path:
                shutil.copy2(video_path, os.path.join(song_dir, os.path.basename(video_path)))
            if banner_path:
                shutil.copy2(banner_path, os.path.join(song_dir, os.path.basename(banner_path)))
            if bg_path:
                shutil.copy2(bg_path, os.path.join(song_dir, os.path.basename(bg_path)))
        
        # Guardamos el archivo .sm
        sm_file_path = os.path.join(song_dir, f"{song_title}.sm")
        with open(sm_file_path, "w", encoding='utf-8') as f:
            f.write(sm_content)
        
        return song_dir

def main():
    """
    Función principal que maneja la interacción con el usuario y el flujo del programa.
    """
    try:
        print("=== Generador de Canciones DDR ===")
        # Obtenemos las rutas de los archivos
        audio_path = input("\nIngresa la ruta del archivo de audio: ").strip().strip('"')
        video_path = input("Ingresa la ruta del video (opcional): ").strip().strip('"') or None
        
        # Verificamos que el archivo de audio existe
        if not os.path.exists(audio_path):
            raise FileNotFoundError("El archivo de audio no existe")
        
        if video_path and not os.path.exists(video_path):
            print("Advertencia: El archivo de video no existe. Continuando sin video.")
            video_path = None
        
        # Iniciamos el generador con manejo de errores
        print("\nInicializando generador...")
        try:
            generator = DDRSongGenerator(audio_path, video_path)
        except Exception as e:
            print(f"Error al inicializar el generador: {str(e)}")
            raise
        
        # Configuramos la generación
        print("\nConfiguración de la canción:")
        song_title = input("Título de la canción: ").strip()
        while not song_title:  # Aseguramos que el título no esté vacío
            print("El título no puede estar vacío.")
            song_title = input("Título de la canción: ").strip()
        
        # Solicitamos la dificultad con validación
        valid_difficulties = ['beginner', 'easy', 'normal', 'hard', 'all']
        difficulty = input("Dificultad (beginner/easy/normal/hard/all): ").strip().lower() or "normal"
        while difficulty not in valid_difficulties:
            print(f"Dificultad no válida. Opciones: {', '.join(valid_difficulties)}")
            difficulty = input("Dificultad: ").strip().lower() or "normal"
        
        # Generamos los archivos con manejo de errores
        try:
            output_dir = generator.create_stepmania_files(song_title, difficulty)
            print(f"\n¡Archivos generados exitosamente!")
            print(f"Ubicación: {output_dir}")
            print("\nPasos a seguir:")
            print("1. Abre StepMania")
            print("2. La canción aparecerá en el menú de selección")
            print(f"3. Selecciona la dificultad deseada: {difficulty}")
            
        except Exception as e:
            print(f"\nError al generar los archivos: {str(e)}")
            traceback.print_exc()
            raise
        
    except KeyboardInterrupt:
        print("\n\nOperación cancelada por el usuario.")
    except Exception as e:
        print(f"\nError inesperado: {str(e)}")
        traceback.print_exc()
    finally:
        input("\nPresiona Enter para salir...")

if __name__ == "__main__":
=======
"""
Generator_Stepmania.py
Este script genera archivos para el juego StepMania basándose en el análisis de archivos de audio.
Analiza la música para crear patrones de pasos que coincidan con el ritmo y permite incluir videos de fondo.
"""

# Importaciones básicas del sistema y utilidades
import os
import random
import shutil
import traceback
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

# Intentamos importar las bibliotecas opcionales con manejo de errores elegante
try:
    import cv2
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("Advertencia: moviepy no está disponible. El procesamiento de video será limitado.")

try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Error: librosa es necesario para el análisis musical.")
    raise

@dataclass
class DanceStep:
    """
    Representa un paso de baile con sus características específicas.
    
    Attributes:
        direction: Dirección del paso ('left', 'right', 'up', 'down', 'none', o versiones 'hold_')
        timing: Momento exacto del paso en segundos
        duration: Duración del paso en segundos
        beat_strength: Intensidad del beat musical asociado (0.0 a 1.0)
    """
    direction: str
    timing: float
    duration: float
    beat_strength: float

class DDRSongGenerator:
    """
    Clase principal para generar archivos de canciones para StepMania/DDR.
    Esta clase maneja todo el proceso de análisis musical y generación de pasos.
    """
    
    def __init__(self, audio_path: str, video_path: Optional[str] = None, stepmania_path: Optional[str] = None):
        """
        Inicializa el generador con los archivos necesarios y analiza la música.
        
        Args:
            audio_path: Ruta al archivo de audio principal
            video_path: Ruta opcional al archivo de video de fondo
            stepmania_path: Ruta opcional a la instalación de StepMania
        """
        # Verificamos que el archivo de audio existe
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"No se encontró el archivo de audio: {audio_path}")

        # Verificación inicial de video
        if video_path and not MOVIEPY_AVAILABLE:
            print("Advertencia: Se proporcionó video pero moviepy no está disponible.")
            print("El video no será procesado. Instale moviepy para habilitar esta función.")
            video_path = None
        elif video_path and not os.path.exists(video_path):
            print(f"Advertencia: No se encontró el archivo de video: {video_path}")
            video_path = None

        # Configuración de rutas
        self.audio_path = os.path.abspath(audio_path)
        self.video_path = os.path.abspath(video_path) if video_path else None
        self.stepmania_path = stepmania_path or self._find_stepmania_path()

        # Análisis musical inicial
        print("Analizando audio...")
        try:
            self.y, self.sr = librosa.load(audio_path)
            self.tempo, self.beat_frames = librosa.beat.beat_track(y=self.y, sr=self.sr)
            self.beat_times = librosa.frames_to_time(self.beat_frames, sr=self.sr)
            print(f"Tempo detectado: {self.tempo:.1f} BPM")
        except Exception as e:
            print(f"Error en el análisis de audio: {str(e)}")
            print("Usando configuración predeterminada...")
            self.sr = 44100
            self.tempo = 120
            duration = os.path.getsize(audio_path) / (self.sr * 2)
            self.beat_times = np.arange(0, duration, 60.0 / self.tempo)

        # Análisis avanzado
        self.stops = self._analyze_music_stops()
        self._initialize_difficulty_settings()
        
        # Lista para almacenar los pasos generados
        self.steps: List[DanceStep] = []

    def _find_stepmania_path(self) -> Optional[str]:
        """
        Busca la instalación de StepMania en ubicaciones comunes del sistema.
        Si no encuentra una instalación, crea una carpeta temporal para desarrollo.
        """
        possible_paths = [
            "C:/Program Files/Stepmania 5",
            "C:/Program Files (x86)/Stepmania 5",
            "D:/StepMania",
            "D:/Games/Stepmania",
            "E:/Games/Stepmania",
            "./Stepmania",  # Ruta relativa para desarrollo
            "../Stepmania"  # Ruta relativa para desarrollo
        ]
        
        # Buscamos en las rutas predefinidas
        for path in possible_paths:
            if os.path.exists(path):
                print(f"StepMania encontrado en: {path}")
                return path

        # Si no encontramos StepMania, creamos una carpeta temporal
        temp_path = "./StepmaniaTemp"
        os.makedirs(temp_path, exist_ok=True)
        print(f"No se encontró StepMania. Usando carpeta temporal: {temp_path}")
        return temp_path

    def _initialize_difficulty_settings(self):
        """
        Configura los parámetros específicos para cada nivel de dificultad.
        Define patrones de pasos, probabilidades y configuraciones para cada nivel.
        """
        self.difficulty_settings = {
            'beginner': {
                'level': 2,
                'ratings': '0.135,0.213,0.036,0.081,0.000',
                'max_steps_per_beat': 1,
                'holds_probability': 0.1,
                'empty_measures_intro': 2,
                'patterns': [['left'], ['right'], ['up'], ['down']]
            },
            'easy': {
                'level': 4,
                'ratings': '0.263,0.319,0.108,0.009,0.000',
                'max_steps_per_beat': 1,
                'holds_probability': 0.15,
                'empty_measures_intro': 1,
                'patterns': [['left', 'right'], ['up', 'down']]
            },
            'normal': {
                'level': 6,
                'ratings': '0.514,0.532,0.189,0.379,0.000',
                'max_steps_per_beat': 2,
                'holds_probability': 0.2,
                'empty_measures_intro': 1,
                'patterns': [['left', 'right', 'up'], ['down', 'left', 'right']]
            },
            'hard': {
                'level': 9,
                'ratings': '0.740,0.710,0.523,0.216,0.054',
                'max_steps_per_beat': 3,
                'holds_probability': 0.25,
                'empty_measures_intro': 0,
                'patterns': [['left', 'right', 'up', 'down'], ['up', 'down', 'left', 'right']]
            }
        }


    def _analyze_music_stops(self) -> List[Dict[str, float]]:
        """
        Detecta momentos de pausa significativa en la música usando un análisis más selectivo.
        Solo detecta pausas en cambios muy dramáticos de intensidad y usa duraciones más cortas.
        """
        stops = []
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        
        # Calculamos un umbral más exigente
        mean_strength = np.mean(onset_env)
        std_strength = np.std(onset_env)
        threshold = mean_strength - (std_strength * 2)  # Más selectivo
        
        # Buscamos segmentos con caídas significativas de intensidad
        window_size = 4  # Ventana para analizar el contexto
        for i in range(window_size, len(onset_env) - window_size):
            # Verificamos si hay una caída significativa
            before = np.mean(onset_env[i-window_size:i])
            current = onset_env[i]
            after = np.mean(onset_env[i+1:i+1+window_size])
            
            if (before > threshold and 
                current < threshold and 
                after > threshold):
                
                time = librosa.frames_to_time(i, sr=self.sr)
                
                # Calculamos una duración más apropiada basada en el BPM
                beat_duration = 60.0 / self.tempo
                stop_duration = beat_duration * 0.25  # 1/4 de beat
                
                stops.append({
                    'time': time,
                    'duration': min(0.200, stop_duration)  # Máximo 200ms
                })
        
        # Filtramos pausas muy cercanas
        filtered_stops = []
        last_time = -1
        min_interval = 60.0 / self.tempo  # Un beat de separación
        
        for stop in stops:
            if last_time == -1 or (stop['time'] - last_time) >= min_interval:
                filtered_stops.append(stop)
                last_time = stop['time']
        
        return filtered_stops
    
    # def _analyze_music_stops(self) -> List[Dict[str, float]]:
    #     """
    #     Detecta momentos de pausa significativa en la música usando análisis de onsets.
    #     Estas paradas se utilizarán para generar momentos de pausa en los pasos.
    #     """
    #     stops = []
    #     onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        
    #     # Buscamos puntos donde la intensidad cae significativamente
    #     threshold = np.mean(onset_env) * 0.3
    #     for i in range(1, len(onset_env)):
    #         if onset_env[i] < threshold and onset_env[i-1] > threshold:
    #             time = librosa.frames_to_time(i, sr=self.sr)
    #             stops.append({
    #                 'time': time,
    #                 'duration': 0.620  # Duración estándar de pausa
    #             })
        
    #     return stops

    def _analyze_beat_strength(self) -> List[Dict[str, float]]:
        """
        Realiza un análisis detallado de las características musicales de cada beat.
        Esto permite sincronizar los pasos con diferentes elementos de la música.
        """
        print("Analizando intensidad de beats...")
        beat_info = []
        hop_length = 512
        
        # Análisis del espectrograma mel para diferentes bandas de frecuencia
        spec = librosa.feature.melspectrogram(
            y=self.y, 
            sr=self.sr,
            n_mels=128,
            hop_length=hop_length
        )
        
        # Detección de onsets para ritmo
        onset_env = librosa.onset.onset_strength(
            y=self.y, 
            sr=self.sr,
            hop_length=hop_length,
            aggregate=np.median
        )
        
        # Análisis de contraste espectral para cambios dramáticos
        spectral_contrast = librosa.feature.spectral_contrast(
            y=self.y, 
            sr=self.sr,
            hop_length=hop_length
        )
        
        # Analizamos cada beat individual
        for beat_time in self.beat_times:
            frame = librosa.time_to_frames(beat_time, sr=self.sr, hop_length=hop_length)
            if frame < len(onset_env):
                strength = onset_env[frame]
                
                if frame < spec.shape[1]:
                    spec_beat = spec[:, frame]
                    # Analizamos diferentes rangos de frecuencia
                    bass_strength = np.mean(spec_beat[:20])    # Bajos
                    mid_strength = np.mean(spec_beat[20:80])   # Medios
                    high_strength = np.mean(spec_beat[80:])    # Altos
                    contrast = np.mean(spectral_contrast[:, frame])
                else:
                    bass_strength = mid_strength = high_strength = contrast = 0.0
                
                beat_info.append({
                    'time': float(beat_time),
                    'strength': float(strength),
                    'bass_strength': float(bass_strength),
                    'mid_strength': float(mid_strength),
                    'high_strength': float(high_strength),
                    'contrast': float(contrast)
                })
        
        return beat_info
    
    def generate_sequence(self, difficulty: str = 'normal') -> List[DanceStep]:
        """
        Genera la secuencia completa de pasos para toda la duración de la canción.
        
        Args:
            difficulty: Nivel de dificultad deseado ('beginner', 'easy', 'normal', 'hard')
        Returns:
            Lista de pasos de baile sincronizados con la música
        """
        settings = self.difficulty_settings[difficulty.lower()]
        beat_info = self._analyze_beat_strength()
        self.steps = []
        
        # 1. Calculamos la duración exacta y los beats totales
        song_duration = librosa.get_duration(y=self.y, sr=self.sr)
        
        # Manejamos el tempo de manera segura
        try:
            if hasattr(self.tempo, 'item'):
                tempo = float(self.tempo.item())
            else:
                tempo = float(self.tempo)
        except (AttributeError, IndexError, TypeError):
            print("Advertencia: Usando tempo predeterminado")
            tempo = 120.0

        # 2. Calculamos la estructura completa de la canción
        beats_per_second = tempo / 60
        total_beats = int(np.ceil(song_duration * beats_per_second))
        beats_per_measure = 16  # Cada compás tiene 4 beats × 4 subdivisiones
        total_measures = int(np.ceil(total_beats / beats_per_measure))
        
        # Mostramos información de diagnóstico
        print(f"Duración de la canción: {song_duration:.2f} segundos")
        print(f"Tempo: {tempo:.2f} BPM")
        print(f"Total de beats necesarios: {total_beats}")
        print(f"Total de compases a generar: {total_measures}")
        
        # 3. Generamos los compases vacíos iniciales (introducción)
        empty_measures = settings['empty_measures_intro']
        for measure in range(empty_measures):
            for _ in range(beats_per_measure):
                self.steps.append(DanceStep(
                    direction='none',
                    timing=len(self.steps),
                    duration=0.5,
                    beat_strength=0
                ))
        
        # 4. Generamos el resto de los compases
        for measure in range(empty_measures, total_measures):
            measure_start_beat = measure * beats_per_measure
            
            # Generamos los 16 pasos del compás
            steps_in_measure = 0
            while steps_in_measure < beats_per_measure:
                current_beat = measure_start_beat + steps_in_measure
                if current_beat >= total_beats:
                    break
                
                # Calculamos el índice del beat actual para el análisis musical
                beat_idx = min(int((current_beat / total_beats) * len(beat_info)), len(beat_info) - 1)
                beat = beat_info[beat_idx]
                
                # Decidimos si poner un paso o un espacio vacío
                if (beat['strength'] > 0.3 or 
                    beat['bass_strength'] > 0.3 or 
                    steps_in_measure % 8 == 0):  # Aseguramos al menos un paso cada 8 beats
                    
                    # Seleccionamos la dirección del paso
                    direction = random.choice(['left', 'down', 'up', 'right'])
                    
                    # Añadimos el paso
                    self.steps.append(DanceStep(
                        direction=direction,
                        timing=current_beat,
                        duration=0.5,
                        beat_strength=beat['strength']
                    ))
                    steps_in_measure += 1
                    
                    # Añadimos tres espacios vacíos después del paso
                    for _ in range(min(3, beats_per_measure - steps_in_measure)):
                        self.steps.append(DanceStep(
                            direction='none',
                            timing=current_beat + 1,
                            duration=0.5,
                            beat_strength=0
                        ))
                        steps_in_measure += 1
                else:
                    # Añadimos un espacio vacío
                    self.steps.append(DanceStep(
                        direction='none',
                        timing=current_beat,
                        duration=0.5,
                        beat_strength=0
                    ))
                    steps_in_measure += 1
        
        # 5. Verificación final
        print(f"Pasos generados: {len(self.steps)}")
        print(f"Duración esperada: {total_beats / beats_per_second:.2f} segundos")
        print(f"Duración real: {len(self.steps) / (beats_per_second * 4):.2f} segundos")
        
        return self.steps
    
    
    
    # def generate_sequence(self, difficulty: str = 'normal') -> List[DanceStep]:
    #     """
    #     Genera la secuencia de pasos basada en el análisis musical y el nivel de dificultad.
    #     """
    #     settings = self.difficulty_settings[difficulty.lower()]
    #     beat_info = self._analyze_beat_strength()
    #     self.steps = []
        
    #     # Calculamos parámetros básicos de manera segura
    #     song_duration = librosa.get_duration(y=self.y, sr=self.sr)
        
    #     try:
    #         if hasattr(self.tempo, 'item'):
    #             tempo = float(self.tempo.item())
    #         else:
    #             tempo = float(self.tempo)
    #     except (AttributeError, IndexError, TypeError):
    #         print("Advertencia: Usando tempo predeterminado")
    #         tempo = 120.0

    #     beats_per_second = tempo / 60
    #     total_beats = int(song_duration * beats_per_second)
        
    #     if difficulty.lower() == 'beginner':
    #         empty_measures = settings['empty_measures_intro']
    #         current_beat = empty_measures * 4
            
    #         # Generamos los compases vacíos iniciales
    #         for _ in range(current_beat):
    #             self.steps.append(DanceStep(
    #                 direction='none',
    #                 timing=current_beat,
    #                 duration=0.5,
    #                 beat_strength=0
    #             ))
    #             current_beat += 1
        
    #         # Generamos los pasos principales con un ritmo más dinámico
    #         while current_beat < total_beats:
    #             beat_idx = min(int(current_beat * len(beat_info) / total_beats), len(beat_info) - 1)
    #             beat = beat_info[beat_idx]
            
    #             # Reducimos umbral y mejoramos la lógica de generación de pasos
    #             if (beat['strength'] > 0.3 or beat['bass_strength'] > 0.3 or 
    #                 current_beat % 8 == 0):  # Asegura pasos regulares
                    
    #                 direction = random.choice(['left', 'down', 'up', 'right'])
                    
    #                 # Un paso seguido de tres beats vacíos (patrón 1-3)
    #                 self.steps.append(DanceStep(
    #                     direction=direction,
    #                     timing=current_beat,
    #                     duration=0.5,
    #                     beat_strength=beat['strength']
    #                 ))
    #                 current_beat += 1
                    
    #                 # Tres beats vacíos
    #                 for _ in range(3):
    #                     self.steps.append(DanceStep(
    #                         direction='none',
    #                         timing=current_beat,
    #                         duration=0.5,
    #                         beat_strength=0
    #                     ))
    #                     current_beat += 1
    #             else:
    #                 # Beat débil, un solo paso vacío
    #                 self.steps.append(DanceStep(
    #                     direction='none',
    #                     timing=current_beat,
    #                     duration=0.5,
    #                     beat_strength=0
    #                 ))
    #             current_beat += 1
    # def generate_sequence(self, difficulty: str = 'normal') -> List[DanceStep]:
    #     """
    #     Genera la secuencia de pasos basada en el análisis musical y el nivel de dificultad.
    #     """
    #     settings = self.difficulty_settings[difficulty.lower()]
    #     beat_info = self._analyze_beat_strength()
    #     self.steps = []
        
    #     # Calculamos parámetros básicos
    #     song_duration = librosa.get_duration(y=self.y, sr=self.sr)
    #     beats_per_second = float(self.tempo[0] if hasattr(self.tempo, 'item') else self.tempo) / 60
    #     total_beats = int(song_duration * beats_per_second)

    #     if difficulty.lower() == 'beginner':
    #         empty_measures = 2
    #         current_beat = empty_measures * 4

    #     # Secure time handling
    #     try:
    #         if hasattr(self.tempo, 'item'):
    #             tempo = float(self.tempo.item())
    #         else:
    #             tempo = float(self.tempo)
    #     except (AttributeError, IndexError, TypeError):
    #         print("Advertencia: Usando tempo predeterminado")
    #         tempo = 120.0

    #     beats_per_second = tempo / 60
    #     total_beats = int(song_duration * beats_per_second)
        
    #     # Añadimos compases vacíos al inicio para dar tiempo al jugador
    #     empty_measures = settings['empty_measures_intro']
    #     current_beat = empty_measures * 4
        
    #     for _ in range(current_beat):
    #         self.steps.append(DanceStep(
    #             direction='none',
    #             timing=current_beat,
    #             duration=0.5,
    #             beat_strength=0
    #         ))
    #         current_beat += 1
        
    #     # Generamos los pasos principales
    #     while current_beat < total_beats:
    #         beat_idx = min(int(current_beat * len(beat_info) / total_beats), len(beat_info) -1)
    #         beat = beat_info[beat_idx]
                        
    #         if beat['strength'] > 0.3 or beat['bass_strength'] > 0.3:
    #             direction = random.choice(['left', 'down', 'up', 'right'])
                 
    #             self.steps.append(DanceStep(
    #                 direction=direction,
    #                 timing=current_beat,
    #                 duration=0.5,
    #                 beat_strength=beat['strength']
    #             ))
    #             current_beat += 1

    #             for _ in range(3):
    #                 self.steps.append(DanceStep(
    #                     direction='none',
    #                     timing=current_beat,
    #                     duration=0.5,
    #                     beat_strength=0
    #                 ))
    #                 current_beat += 1

    #         else:
    #             # En beats débiles, añadimos paso vacío
    #             self.steps.append(DanceStep(
    #                 direction='none',
    #                 timing=current_beat,
    #                 duration=0.5,
    #                 beat_strength=0
    #             ))
    #             current_beat += 1
        
    #     return self.steps

    def process_background_video(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Procesa el video de fondo para máxima compatibilidad con StepMania 3.9
        """
        if not self.video_path or not MOVIEPY_AVAILABLE:
            return None, None, None
            
        try:
            video = VideoFileClip(self.video_path)
            
            # Ajustamos tamaño para StepMania 3.9
            target_size = (640, 480)
            processed_video = video.resize(target_size)
            
            # Extraemos frames para banner y background
            banner_time = video.duration * 0.25
            bg_time = video.duration * 0.75
            
            banner_frame = video.get_frame(banner_time)
            bg_frame = video.get_frame(bg_time)
                
            # Procesamos los frames
            banner = cv2.resize(banner_frame, (256, 80))
            background = cv2.resize(bg_frame, target_size)
                
            # Preparamos rutas de salida con formato más compatible
            base_path = os.path.splitext(self.video_path)[0]
            video_out = f"{base_path}_processed.mpg"  # Cambiamos a MPEG-1
            banner_out = f"{base_path}-bn.png"
            bg_out = f"{base_path}-bg.png"
            
            # Guardamos el video con configuración más compatible
            processed_video.write_videofile(
                video_out,
                codec='mpeg1video',  # Usamos MPEG-1
                audio=False,
                fps=30,
                bitrate="2000k"  # Bajamos el bitrate para mejor compatibilidad
            )
            
            cv2.imwrite(banner_out, cv2.cvtColor(banner, cv2.COLOR_RGB2BGR))
            cv2.imwrite(bg_out, cv2.cvtColor(background, cv2.COLOR_RGB2BGR))
            
            return video_out, banner_out, bg_out
            
        except Exception as e:
            print(f"Error procesando video: {str(e)}")
            return self.video_path, None, None

    # def process_background_video(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    #     """
    #     Procesa el video de fondo para adaptarlo a los requisitos de StepMania.
    #     Genera el video principal, banner y background.
    #     """
    #     if not self.video_path or not MOVIEPY_AVAILABLE:
    #         return None, None, None
            
    #     try:
    #         video = VideoFileClip(self.video_path)
            
    #         Ajustamos el video al tamaño estándar de StepMania
    #         target_size = (640, 480)
    #         processed_video = video.resize(target_size)
            
    #         Extraemos frames para banner y background
    #         banner_time = video.duration * 0.25
    #         bg_time = video.duration * 0.75
            
    #         banner_frame = video.get_frame(banner_time)
    #         bg_frame = video.get_frame(bg_time)
            
    #         Procesamos los frames
    #         banner = cv2.resize(banner_frame, (256, 80))
    #         background = cv2.resize(bg_frame, target_size)
            
    #         Preparamos rutas de salida
    #         base_path = os.path.splitext(self.video_path)[0]
    #         video_out = f"{base_path}_processed.avi"
    #         banner_out = f"{base_path}-bn.png"
    #         bg_out = f"{base_path}-bg.png"
            
    #         Guardamos los archivos
    #         processed_video.write_videofile(
    #             video_out,
    #             codec='libx264',
    #             audio=False,
    #             fps=30
    #         )
            
    #         cv2.imwrite(banner_out, cv2.cvtColor(banner, cv2.COLOR_RGB2BGR))
    #         cv2.imwrite(bg_out, cv2.cvtColor(background, cv2.COLOR_RGB2BGR))
            
    #         return video_out, banner_out, bg_out
            
    #     except Exception as e:
    #         print(f"Error procesando video: {str(e)}")
    #         return self.video_path, None, None

    def _generate_sm_content(self, title: str, difficulties: List[str]) -> str:
        """
        Genera el contenido del archivo .sm con el formato requerido por StepMania.
        """
        try:
        # Si self.tempo es un array de numpy, obtenemos el primer valor
            if hasattr(self.tempo, 'item'):
                tempo = float(self.tempo.item())
        # Si es un número normal, lo convertimos directamente
            else:
                tempo = float(self.tempo)
        except (AttributeError, IndexError, TypeError):
            # Si algo falla, usamos un valor predeterminado seguro
            print("Advertencia: No se pudo determinar el tempo exacto. Usando valor predeterminado.")
            tempo = 120.0
        
        # Procesamos video si existe
        video_path, banner_path, bg_path = self.process_background_video() if self.video_path else (None, None, None)
        
        # Construimos el encabezado
        audio_filename = os.path.basename(self.audio_path)
        content = f"""#TITLE:{title};
#SUBTITLE:;
#ARTIST:Generated;
#TITLETRANSLIT:;
#SUBTITLETRANSLIT:;
#ARTISTTRANSLIT:;
#GENRE:;
#CREDIT:Generated by Python;
#BANNER:{os.path.basename(banner_path) if banner_path else ''};
#BACKGROUND:{os.path.basename(bg_path) if bg_path else ''};
#LYRICSPATH:;
#CDTITLE:;
#MUSIC:audio{os.path.splitext(audio_filename)[1]};
#OFFSET:-0.100;
#SAMPLESTART:52.840;
#SAMPLELENGTH:13.460;
#SELECTABLE:YES;
#DISPLAYBPM:{tempo:.3f};
#BPMS:0.000={tempo:.3f};"""

        # Añadimos información de paradas musicales
        if self.stops:
            stops_str = ','.join([f"{stop['time']:.3f}={stop['duration']:.3f}" for stop in self.stops])
            content += f"\n#STOPS:{stops_str};"
        else:
            content += "\n#STOPS:;"

        # Configuramos video de fondo
        if video_path:
            video_filename = os.path.basename(video_path)
            content += f"\n#BGCHANGES:-0.950={video_filename}=1.000=1=1=0,\n99999=-nosongbg-=1.000=0=0=0 // don't automatically add -songbackground-\n;"
        else:
            content += "\n#BGCHANGES:;"

        # Generamos notas para cada dificultad
        for diff in difficulties:
            settings = self.difficulty_settings[diff.lower()]
            self.generate_sequence(diff)
            content += f"""
//---------------dance-single - ----------------
#NOTES:
     dance-single:
     :
     {diff.capitalize()}:
     {settings['level']}:
     {settings['ratings']}:
{self._convert_steps_to_measures()}
;"""

        return content

    def _convert_steps_to_measures(self) -> str:
        """
        Convierte los pasos al formato de medidas de StepMania.
        """
        if not self.steps:
            return "0000\n0000\n0000\n0000\n"
        
        BEATS_PER_MEASURE = 4
        SUBDIVISIONS = 4
        measure_str = ""
        
        for i in range(0, len(self.steps), BEATS_PER_MEASURE * SUBDIVISIONS):
            measure_steps = self.steps[i:i + BEATS_PER_MEASURE * SUBDIVISIONS]
            measure_content = ""
            
            for step in measure_steps:
                # Convertimos cada paso al formato de flechas
                if step.direction == 'left':
                    arrow = "1000"
                elif step.direction == 'down':
                    arrow = "0100"
                elif step.direction == 'up':
                    arrow = "0010"
                elif step.direction == 'right':
                    arrow = "0001"
                elif step.direction.startswith('hold_'):
                    base_dir = step.direction.split('_')[1]
                    arrow = self._get_hold_arrow(base_dir)
                else:
                    arrow = "0000"
                
                measure_content += arrow + "\n"
            
            # Completamos la medida si es necesario
            while len(measure_content.split('\n')) < BEATS_PER_MEASURE * SUBDIVISIONS + 1:
                measure_content += "0000\n"
            
            measure_str += measure_content + ",\n"
        
        return measure_str

    def _get_hold_arrow(self, direction: str) -> str:
        """
        Convierte una dirección de flecha sostenida al formato de StepMania.
        """
        hold_map = {
            'left': '2000',
            'down': '0200',
            'up': '0020',
            'right': '0002'
        }
        return hold_map.get(direction, '0000')

    def create_stepmania_files(self, song_title: str, difficulty: str = 'normal') -> str:
        """
        Crea todos los archivos necesarios para la canción en StepMania.
        """
        if not self.stepmania_path:
            raise ValueError("No se pudo encontrar la instalación de StepMania")
        
        # Creamos el directorio para la canción
        songs_path = os.path.join(self.stepmania_path, 'Songs')
        song_dir = os.path.join(songs_path, song_title)
        os.makedirs(song_dir, exist_ok=True)
        
        # Generamos archivos
        difficulties = ['beginner', 'easy', 'normal', 'hard'] if difficulty == 'all' else [difficulty]
        sm_content = self._generate_sm_content(song_title, difficulties)
        
        # Copiamos el archivo de audio
        print("Copiando archivos...")
        audio_ext = os.path.splitext(self.audio_path)[1]
        shutil.copy2(self.audio_path, os.path.join(song_dir, f"audio{audio_ext}"))
        
        # Procesamos y copiamos archivos de video
        if self.video_path:
            video_path, banner_path, bg_path = self.process_background_video()
            if video_path:
                shutil.copy2(video_path, os.path.join(song_dir, os.path.basename(video_path)))
            if banner_path:
                shutil.copy2(banner_path, os.path.join(song_dir, os.path.basename(banner_path)))
            if bg_path:
                shutil.copy2(bg_path, os.path.join(song_dir, os.path.basename(bg_path)))
        
        # Guardamos el archivo .sm
        sm_file_path = os.path.join(song_dir, f"{song_title}.sm")
        with open(sm_file_path, "w", encoding='utf-8') as f:
            f.write(sm_content)
        
        return song_dir

def main():
    """
    Función principal que maneja la interacción con el usuario y el flujo del programa.
    """
    try:
        print("=== Generador de Canciones DDR ===")
        # Obtenemos las rutas de los archivos
        audio_path = input("\nIngresa la ruta del archivo de audio: ").strip().strip('"')
        video_path = input("Ingresa la ruta del video (opcional): ").strip().strip('"') or None
        
        # Verificamos que el archivo de audio existe
        if not os.path.exists(audio_path):
            raise FileNotFoundError("El archivo de audio no existe")
        
        if video_path and not os.path.exists(video_path):
            print("Advertencia: El archivo de video no existe. Continuando sin video.")
            video_path = None
        
        # Iniciamos el generador con manejo de errores
        print("\nInicializando generador...")
        try:
            generator = DDRSongGenerator(audio_path, video_path)
        except Exception as e:
            print(f"Error al inicializar el generador: {str(e)}")
            raise
        
        # Configuramos la generación
        print("\nConfiguración de la canción:")
        song_title = input("Título de la canción: ").strip()
        while not song_title:  # Aseguramos que el título no esté vacío
            print("El título no puede estar vacío.")
            song_title = input("Título de la canción: ").strip()
        
        # Solicitamos la dificultad con validación
        valid_difficulties = ['beginner', 'easy', 'normal', 'hard', 'all']
        difficulty = input("Dificultad (beginner/easy/normal/hard/all): ").strip().lower() or "normal"
        while difficulty not in valid_difficulties:
            print(f"Dificultad no válida. Opciones: {', '.join(valid_difficulties)}")
            difficulty = input("Dificultad: ").strip().lower() or "normal"
        
        # Generamos los archivos con manejo de errores
        try:
            output_dir = generator.create_stepmania_files(song_title, difficulty)
            print(f"\n¡Archivos generados exitosamente!")
            print(f"Ubicación: {output_dir}")
            print("\nPasos a seguir:")
            print("1. Abre StepMania")
            print("2. La canción aparecerá en el menú de selección")
            print(f"3. Selecciona la dificultad deseada: {difficulty}")
            
        except Exception as e:
            print(f"\nError al generar los archivos: {str(e)}")
            traceback.print_exc()
            raise
        
    except KeyboardInterrupt:
        print("\n\nOperación cancelada por el usuario.")
    except Exception as e:
        print(f"\nError inesperado: {str(e)}")
        traceback.print_exc()
    finally:
        input("\nPresiona Enter para salir...")

if __name__ == "__main__":
>>>>>>> 19c6aee735d57b54c50271585000b9113ca08fdc
    main()