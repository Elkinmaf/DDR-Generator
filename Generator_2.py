import os
import random
import shutil
import traceback
from dataclasses import dataclass
from typing import List, Dict, Any

import librosa
import numpy as np

@dataclass
class DanceStep:
    """Representa un paso de baile con sus características."""
    direction: str
    timing: float
    duration: float
    beat_strength: float

class DDRSongGenerator:
    """Clase para generar archivos de canciones para StepMania/DDR."""
    
    def __init__(self, audio_path: str, video_path: str = None, stepmania_path: str = None):
        """
        Inicializa el generador de canciones DDR.
        
        :param audio_path: Ruta del archivo de audio
        :param video_path: Ruta opcional del archivo de video de fondo
        :param stepmania_path: Ruta opcional de instalación de StepMania
        """
        # Rutas de archivos
        self.audio_path = os.path.abspath(audio_path)
        self.video_path = os.path.abspath(video_path) if video_path else None
        self.stepmania_path = stepmania_path or self._find_stepmania_path()
        
        # Análisis del audio
        print("Analizando audio...")
        self.y, self.sr = librosa.load(audio_path)
        self.tempo, self.beat_frames = librosa.beat.beat_track(y=self.y, sr=self.sr)
        self.beat_times = librosa.frames_to_time(self.beat_frames, sr=self.sr)
        
        # Configuración de pasos
        self.directions = {
            'up': '2',
            'down': '1',
            'left': '3',
            'right': '4',
            'none': '0'
        }
        
        # Patrones predefinidos por dificultad
        self.patterns = {
            'beginner': [
                ['left'], ['right'], ['up'], ['down']  # Solo pasos individuales
            ],
            'easy': [
                ['left', 'right'], ['up', 'down'],
                ['left', 'up'], ['right', 'down']  # Combinaciones simples
            ],
            'normal': [
                ['left', 'right', 'up'],
                ['down', 'left', 'right'],
                ['up', 'down', 'left']  # Patrones de 3 pasos
            ],
            'hard': [
                ['left', 'right', 'up', 'down'],
                ['up', 'down', 'left', 'right'],
                ['left', 'up', 'right', 'down']  # Patrones complejos
            ]
        }
        
        self.steps: List[DanceStep] = []

    def _find_stepmania_path(self) -> str:
        """
        Busca la instalación de StepMania en ubicaciones comunes.
        
        :return: Ruta de instalación de StepMania o None si no se encuentra
        """
        possible_paths = [
            "C:/Program Files/Stepmania 5",
            "C:/Program Files (x86)/Stepmania 5",
            "D:/StepMania",
            "D:/Games/Stepmania",
            "E:/Games/Stepmania",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

    def _analyze_beat_strength(self) -> List[Dict[str, float]]:
        """
        Analiza las características musicales de cada beat con mayor precisión.
        """
        print("Analizando intensidad de beats...")
        beat_info = []
        
        # Analizar el espectro completo
        hop_length = 512
        
        spec = librosa.feature.melspectrogram(
            y=self.y, 
            sr=self.sr,
            n_mels=128,
            hop_length=hop_length
        )
        
        # Detectar ritmo con más detalle
        onset_env = librosa.onset.onset_strength(
            y=self.y, 
            sr=self.sr,
            hop_length=hop_length,
            aggregate=np.median
        )

        spectral_contrast = librosa.feature.spectral_contrast(
            y=self.y,
            sr=self.sr,
            hop_length=hop_length
        )

        for beat_time in self.beat_times:
            frame = librosa.time_to_frames(beat_time, sr=self.sr, hop_length=hop_length)
            if frame < len(onset_env):
                # Características básicas
                strength = onset_env[frame]
                
                # Analizar bandas de frecuencia
                if frame < spec.shape[1]:
                    spec_beat = spec[:, frame]
                    bass_strength = np.mean(spec_beat[:20])  # Frecuencias bajas
                    mid_strength = np.mean(spec_beat[20:80])  # Frecuencias medias
                    high_strength = np.mean(spec_beat[80:])  # Frecuencias altas
                else:
                    bass_strength = mid_strength = high_strength = 0.0
                
                contrast = np.mean(spectral_contrast[:, frame])

                beat_info.append({
                    'strength': float(strength),
                    'bass_strength': float(bass_strength),
                    'mid_strength': float(mid_strength),
                    'high_strength': float(high_strength),
                    'contrast': float(contrast),
                    'time':float(beat_time)
                })
        
        return beat_info

    def generate_sequence(self, difficulty: str = 'normal') -> List[DanceStep]:

        if self.tempo is None or (hasattr(self.tempo, 'size') and self.tempo.size == 0):
            
            default_tempo = 120.0
            print(f"No se pudo detectar el tempo. Usando valor predeterminado: {default_tempo} BPM")
            beats_per_second = default_tempo / 60
        else:
            beats_per_second = float(self.tempo[0] if hasattr(self.tempo, 'item') else self.tempo) / 60





        song_duration = librosa.get_duration(y=self.y, sr=self.sr)
        beats_per_second = float(self.tempo[0] if hasattr(self.tempo, 'item') else self.tempo) / 60

        total_beats = int(song_duration * beats_per_second)

        print(f"Tempo detectado: {beats_per_second * 60:.1f} BPM")
        print(f"Duración de la canción: {song_duration:.1f} segundos")
        print(f"Total de beats: {total_beats}")

        print(f"Generando secuencia para dificultad: {difficulty}")
        beat_info = self._analyze_beat_strength()

        if difficulty.lower() == 'beginner':
            self.steps = []
            current_beat = 0

            for _ in range(8):
                self.steps.append(DanceStep(
                    direction='none',
                    timing=current_beat,
                    duration=0.5,
                    beat_strength=0
                ))
                current_beat +=1

            basic_directions = ['left', 'down', 'up', 'right']
            last_direction = None

        # Assure to generate enough beats
        while current_beat < total_beats:

            if current_beat % 16 == 0:
                for _ in range(4):
                    self.steps.append(DanceStep(
                        direction='none',
                        timing=current_beat,
                        duration=0.5,
                        beat_strength=0
                    ))
                    current_beat +=1
                continue

            available_directions = [d for d in basic_directions if d != last_direction]
            direction = random.choice(available_directions)
            last_direction = direction


            # 10% of probability of hold note
            if random.random() < 0.1:
                self.steps.append(DanceStep(
                    direction=f"hold_{direction}",
                    timing=current_beat,
                    duration=2.0,
                    beat_strength=1.0
                ))
                current_beat += 2

                for _ in range(6): # 6 empty beats after hold
                    self.steps.append(DanceStep(
                        direction='none',
                        timing=current_beat,
                        duration=0.5,
                        beat_strength=0
                    ))
                       
                    current_beat += 1

            else:
                    #Normal step
                self.steps.append(DanceStep(
                    direction=direction,
                    timing=current_beat,
                    duration=0.5,
                    beat_strength=0.5
                ))
                current_beat += 1

                # 3 empty beats after step   
                for _ in range(3):
                    self.steps.append(DanceStep(
                        direction='none',
                        timing=current_beat,
                        duration=0.5,
                        beat_strength=0
                    ))
                    current_beat += 1
        else:
            difficulty_settings = {
                'easy': {
                    'max_steps_per_beat': 1,
                    'strength_threshold': 0.6,
                    'bass_threshold': 0.5
                },
                'normal': {
                    'max_steps_per_beat': 2,
                    'strength_threshold': 0.5,
                    'bass_threshold': 0.4
                },
                'hard': {
                    'max_steps_per_beat': 3,
                    'strength_threshold': 0.4,
                    'bass_threshold': 0.3
                }
            }
        
            settings = difficulty_settings.get(difficulty.lower(), difficulty_settings['normal'])
            patterns = self.patterns.get(difficulty.lower(), self.patterns['normal'])
        
            self.steps = []  # Reiniciar pasos
        
        # Generar pasos basados en el análisis
            for i, beat in enumerate(beat_info):
                strength = beat['strength']
                bass_strong = beat['bass_strength'] > settings['bass_threshold']
            
                if strength > settings['strength_threshold'] or bass_strong:
                # Seleccionar un patrón
                    pattern = random.choice(patterns)
                
                # Calcular duración
                if i < len(self.beat_times) - 1:
                    total_duration = self.beat_times[i + 1] - self.beat_times[i]
                else:
                    total_duration = 0.5
                
                step_duration = total_duration / len(pattern)
                
                # Generar pasos del patrón
                for step_idx, direction in enumerate(pattern):
                    step = DanceStep(
                        direction=direction,
                        timing=self.beat_times[i] + (step_idx * step_duration),
                        duration=step_duration,
                        beat_strength=strength
                    )
                    self.steps.append(step)
        
            return self.steps

    def _get_sm_direction(self, direction: str) -> str:
        """Convierte una dirección a formato SM"""
        direction_map = {
            'left': '1000',
            'down': '0100',
            'up': '0010',
            'right': '0001',
            'none': '0000',
            'hold_left': '2000',
            'hold_down': '0200',
            'hold_up': '0020',
            'hold_right': '0002'
        }
        return direction_map.get(direction, '0000')
    
    def create_stepmania_files(self, song_title: str, difficulty: str = 'normal') -> str:
        """
        Crea todos los archivos necesarios en la carpeta de StepMania.
        
        :param song_title: Título de la canción
        :param difficulty: Nivel de dificultad o 'all' para generar múltiples
        :return: Ruta del directorio de la canción generada
        """
        if not self.stepmania_path:
            raise ValueError("No se pudo encontrar la instalación de Stepmania")
        
        # Crear directorio para la canción
        songs_path = os.path.join(self.stepmania_path, 'Songs')
        song_dir = os.path.join(songs_path, song_title)
        os.makedirs(song_dir, exist_ok=True)
        
        # Generar archivo .sm con múltiples dificultades
        difficulties = ['beginner', 'easy', 'normal', 'hard'] if difficulty == 'all' else [difficulty]
        sm_content = self._generate_sm_content(song_title, difficulties)
        
        sm_file_path = os.path.join(song_dir, f"{song_title}.sm")
        with open(sm_file_path, "w", encoding='utf-8') as f:
            f.write(sm_content)
        
        # Copiar archivos de audio y video
        print("Copiando archivos...")
        audio_ext = os.path.splitext(self.audio_path)[1]
        shutil.copy2(self.audio_path, os.path.join(song_dir, f"audio{audio_ext}"))
        
        if self.video_path:
            video_ext = os.path.splitext(self.video_path)[1]
            shutil.copy2(self.video_path, os.path.join(song_dir, f"bg{video_ext}"))
        
        return song_dir

    def _generate_sm_content(self, title: str, difficulties: List[str]) -> str:
        """
        Genera el contenido del archivo .sm con múltiples dificultades.
        
        :param title: Título de la canción
        :param difficulties: Lista de dificultades a generar
        :return: Contenido del archivo .sm
        """
        # Convertir tempo a un valor escalar
        tempo = float(self.tempo[0] if hasattr(self.tempo, 'item') else self.tempo)
        
        song_length = librosa.get_duration(y=self.y, sr=self.sr)
        
        # Encabezado del archivo
        audio_filename = os.path.basename(self.audio_path)
        video_line = ""
        if self.video_path:
            video_filename = os.path.basename(self.video_path)
            video_line = f"""#BGCHANGES:-0.000={video_filename}=1.000=1=0=1, 9999=-nosongbg-=1.000=0=0=0 // don't automatically add - songbackground-;"""
        else:
            video_line = "#BGCHANGES:;"

        content = f"""#TITLE:{title};
#SUBTITLE:;
#ARTIST:Generated;
#TITLETRANSLIT:;
#SUBTITLETRANSLIT:;
#ARTISTTRANSLIT:;
#GENRE:;
#CREDIT:Generated by Python;
#BANNER:;
#BACKGROUND:;
#LYRICSPATH:;
#CDTITLE:;
#MUSIC:audio{os.path.splitext(audio_filename)[1]};
#OFFSET:-0.100;
#SAMPLESTART:52.840;
#SAMPLELENGTH:13.460;
#SELECTABLE:YES;
#DISPLAYBPM:{tempo:.3f};
#BPMS:0.000={tempo:.3f};
#STOPS:;
{video_line}\n"""

        # Generar secciones de notas para cada dificultad
        difficulty_levels = {
            'beginner': 2,
            'easy': 4,
            'normal': 6,
            'hard': 8
        }
        
        for diff in difficulties:
            print(f"Generando pasos para dificultad: {diff}")
            self.generate_sequence(diff)
            measures = self._convert_steps_to_measures()
            
            content += f"""
#NOTES:
     dance-single:
     :
     {diff.capitalize()}:
     {difficulty_levels[diff]}:
     0.000,0.000,0.000,0.000,0.000:
{measures}
;"""

        return content

    def _convert_steps_to_measures(self) -> str:
        """
        Convierte los pasos a formato de medidas de StepMania.
        
        :return: Cadena de medidas en formato StepMania
        """
        if not self.steps:
            return "0000\n0000\n0000\n0000\n"
        
        BEATS_PER_MEASURE = 4
        SUBDIVISIONS = 4
        total_measures = int(np.ceil(len(self.steps) / (BEATS_PER_MEASURE * SUBDIVISIONS)))
        
        measures_str = ""
        current_step = 0
        
        for measure in range(total_measures):
            measure_str = ""
            for beat in range(BEATS_PER_MEASURE * SUBDIVISIONS):
                if current_step < len(self.steps):
                    step = self.steps[current_step]
                    arrow = "0000"
                    
                    # Convertir la dirección a formato de flechas
                    if step.direction == 'left':
                        arrow = "1000"
                    elif step.direction == 'down':
                        arrow = "0100"
                    elif step.direction == 'up':
                        arrow = "0010"
                    elif step.direction == 'right':
                        arrow = "0001"
                    
                    measure_str += arrow + "\n"
                    current_step += 1
                else:
                    measure_str += "0000\n"
            
            measures_str += measure_str + ",\n"
        
        return measures_str

def process_background_video(self, video_path: str) -> str:     
    """
    Process the video to assure compatibility with Stepmania.

    Args:
        video_path: Path to the original video

    Returns:
        str: Path to the processed video
    """
    try:
        import cv2
        from moviepy.editor import VideoFileClip

        # Upload the video
        video = VideoFileClip(video_path)

        # Adjust video size (640*480)
        target_size = (640, 480)
        processed_video = video.resize(target_size)

        # Assure that the video has the same length as the music
        song_duration = librosa.get_duration(y=self.y, sr=self.sr)
        if video.duration > song_duration:
            processed_video = processed_video.subclip(0, song_duration)

        # Save the processed video
        output_path = os.path.splitext(video_path)[0] + "_processed.mp4"
        processed_video.write_videofile(
            output_path,
            codec='libx264',
            audio=False,
            fps=30
        )

        return output_path

    except ImportError:
        print("Por favor instala opencv-python y moviepy para procesar videos")
        return video_path

def main():
    """Función principal para ejecutar el generador de canciones DDR."""
    try:
        print("=== Generador de Canciones DDR ===")
        
        # Configuración de rutas - limpiando las comillas
        audio_path = input("\nIngresa la ruta del archivo de audio: ").strip().strip('"').strip("'")
        video_path = input("Ingresa la ruta del archivo de video (opcional, presiona Enter para omitir): ").strip().strip('"').strip("'") or None
          
        
        # Crear generador
        print("\nInicializando generador...")
        generator = DDRSongGenerator(audio_path, video_path)
        
        # Generar archivos
        print("\nGenerando archivos...")
        song_title = input("\nTítulo de la canción: ")
        difficulty = input("Dificultad (beginner/easy/normal/hard/all): ") or "normal"
        
        output_dir = generator.create_stepmania_files(song_title, difficulty)
        
        print(f"\nArchivos generados en: {output_dir}")

    except Exception as e:
        print(f"\nError: {str(e)}")
        traceback.print_exc()
    
    input("\nPresiona Enter para salir...")

if __name__ == "__main__":
    main()