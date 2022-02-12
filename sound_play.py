from pygame import mixer
import time
# Starting the mixer
mixer.init()
  
# Loading the song
mixer.music.load("my.wav")
  
# Setting the volume
mixer.music.set_volume(0.7)
  
# Start playing the song
mixer.music.play()
start_time = time.time()

# infinite loop
while True:
      
    print("Press 'p' to pause, 'r' to resume")
    print("Press 'e' to exit the program")
    # query = input("  ")
    fps = (time.time() - start_time)
    print(fps)
    if fps>3:
        # Stop the mixer
        mixer.music.stop()
        break