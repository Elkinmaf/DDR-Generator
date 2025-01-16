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
        Analiza las características musicales de cada beat.
        
        :return: Lista de diccionarios con información de cada beat
        """
        print("Analizando intensidad de beats...")
        beat_info = []
        
        # Analizar el espectro completo
        spec = librosa.feature.melspectrogram(y=self.y, sr=self.sr)
        
        # Detectar onsets con más detalle
        onset_env = librosa.onset.onset_strength(
            y=self.y, 
            sr=self.sr,
            hop_length=512,
            aggregate=np.median
        )
        
        for beat_time in self.beat_times:
            frame = librosa.time_to_frames(beat_time, sr=self.sr)
            if frame < len(onset_env):
                # Características básicas
                strength = onset_env[frame]
                
                # Analizar bandas de frecuencia
                if frame < spec.shape[1]:
                    spec_beat = spec[:, frame]
                    bass_strength = np.mean(spec_beat[:10])  # Frecuencias bajas
                    mid_strength = np.mean(spec_beat[10:30])  # Frecuencias medias
                    high_strength = np.mean(spec_beat[30:])  # Frecuencias altas
                else:
                    bass_strength = mid_strength = high_strength = 0.0
                
                beat_info.append({
                    'strength': float(strength),
                    'bass_strength': float(bass_strength),
                    'mid_strength': float(mid_strength),
                    'high_strength': float(high_strength)
                })
        
        return beat_info

    def generate_sequence(self, difficulty: str = 'normal') -> List[DanceStep]:
        """
        Genera la secuencia de pasos basada en el análisis musical.
        
        :param difficulty: Nivel de dificultad de los pasos
        :return: Lista de pasos de baile generados
        """
        print(f"Generando secuencia para dificultad: {difficulty}")
        beat_info = self._analyze_beat_strength()
        
        difficulty_settings = {
            'beginner': {
                'max_steps_per_beat': 1,
                'strength_threshold': 0.7,
                'bass_threshold': 0.6
            },
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
        tempo = float(self.tempo) if hasattr(self.tempo, 'item') else self.tempo
        
        song_length = librosa.get_duration(y=self.y, sr=self.sr)
        
        # Encabezado del archivo
        audio_filename = os.path.basename(self.audio_path)
        video_line = ""
        if self.video_path:
            video_filename = os.path.basename(self.video_path)
            video_line = f"#BGCHANGES:bg{os.path.splitext(video_filename)[1]}=0.000=1=1=1=1;"
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
#OFFSET:0.000;
#SAMPLESTART:0.000;
#SAMPLELENGTH:10.000;
#SELECTABLE:YES;
#DISPLAYBPM:*;
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

def main():
    """Función principal para ejecutar el generador de canciones DDR."""
    try:
        print("=== Generador de Canciones DDR ===")
        
        # Configuración de rutas - limpiando las comillas
        audio_path = input("\nIngresa la ruta del archivo de audio: ").strip().strip('"').strip("'")
        video_path = input("Ingresa la ruta del archivo de video (opcional, presiona Enter para omitir): ").strip().strip('"').strip("'") or None
        stepmania_path = input("Ingresa la ruta de instalación de Stepmania (opcional, presiona Enter para búsqueda automática): ").strip().strip('"').strip("'") or None
    
        
        # Crear generador
        print("\nInicializando generador...")
        song_generator = DDRSongGenerator(
            audio_path=audio_path,
            video_path=video_path,
            stepmania_path=stepmania_path
        )
        
        # Configuración de la canción
        song_title = input("\nIngresa el título de la canción: ")
        print("\nDificultades disponibles: beginner, easy, normal, hard, all (para generar todas)")
        difficulty = input("Selecciona la dificultad (default: normal): ").strip() or "normal"
        
        # Generar archivos
        print("\nGenerando archivos...")
        output_dir = song_generator.create_stepmania_files(song_title, difficulty)
        
        print("\n¡Archivos generados exitosamente!")
        print(f"Los archivos se han guardado en: {output_dir}")
        print("\nPara jugar:")
        print("1. Inicia Stepmania")
        print("2. La canción aparecerá en el menú de selección")
        print(f"3. Selecciona la dificultad deseada")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    input("\nPresiona Enter para salir...")

if __name__ == "__main__":
    main()