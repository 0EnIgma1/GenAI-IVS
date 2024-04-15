import gradio as gr
import google.generativeai as genai
import cv2
from PIL import Image
import math
import os
from gtts import gTTS
from playsound import playsound
from PIL import PngImagePlugin
import time
import config

gemini_API = os.environ["gemini_API"] = config.gemini_API
genai.configure(api_key=gemini_API)

vision_model = genai.GenerativeModel('gemini-pro-vision')
text_model = genai.GenerativeModel('gemini-pro')

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

def caption_generation(image, prompt_type):
  prompt = "Explain the scenario or what is happening in the image. Don't mention image in the generated text. No explicit content"
  prompt2 = "Explain the scenario or what is happening in the CCTV image. the explanation should highlight the key events, objects in the environment and potential hazards in the image"
  if (prompt_type == "CCTV_surveillance"):
    prompt = prompt2
  response = vision_model.generate_content([f"{prompt}", image], stream=True, safety_settings=safety_settings)
  response.resolve()
  gen_caption = response.text
  return gen_caption

def split_frames(video, prompt_type):

  cap = cv2.VideoCapture(video)
  local_captions = []
  fps = cap.get(cv2.CAP_PROP_FPS)
  print(f"FPS : {fps}")

  #interval = math.ceil(fps/2)
  #print(f"interval : {interval}")

  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  duration = int(frame_count/fps)

  if (duration <= 6):
    counter = 1
  elif (duration > 7 and duration <=10):
    counter = duration / 5
  else:
    counter = 2
  frame_interval = int(counter * fps)

  frame_count = 0
  extracted_frames_count = 0
  extracted_frames = []

  while True:
      ret, frame = cap.read()
      if not ret:
          break

      if frame_count % frame_interval == 0:
          #cv2.imshow("extracted frame", frame)
          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          img = Image.fromarray(frame_rgb)
          gen_caption = caption_generation(img, prompt_type)
          local_captions.append(gen_caption)
          extracted_frames.append(frame_rgb)

          extracted_frames_count += 1
      frame_count += 1

  #print(local_captions)
  #print(f"extracted frames : {extracted_frames}")
  return local_captions, extracted_frames_count, extracted_frames

#prompt1:"Explain the scenario of what is happening based on the input captions given like a summary. All the captions are sequential and all captions are generated from same video. Combine all the captions summarize them"
def condensation(local_captions, prompt_type):
  prompt1 = "Explain the scenario of what is happening based on the input captions given like a summary. All the captions are sequential and all captions are generated from images from the same video. Combine all the captions summarize them. No explicit content" 
  prompt2 = "Explain the scenario of what is happening by combining the given captions like a summary. All the captions are sequential in order,related to each other and all captions are generated from images from the same video. The summary should be poetic.No explicit content"
  prompt3 = "Explain the scenario of what is happening by combining the given captions like a sports commentary summary. All the captions are sequential in order,related to each other and all captions are generated from images from the same video. Combine all the captions and create a summary. The summary should be like a sports commentary.No explicit content"
  prompt4 = "Explain the scenario of what is happening by combining the given captions like a crisp summary. All the captions are sequential in order,related to each other and all captions are generated from images from the same video. The summary should be crisp summary of all captions combined. Should highlight key events, actions, objects in the environment, potential hazards"
  if (prompt_type == "sports commentary"):
    prompt = prompt3
  elif (prompt_type == "poetic summary"):
    prompt = prompt2
  elif (prompt_type == "CCTV_surveillance"):
    prompt = prompt4
  else:
    prompt = prompt1
  response = text_model.generate_content(f"{prompt}. {local_captions}", stream=True, safety_settings=safety_settings)
  response.resolve()
  return response.text

def video_understanding(video):
  local_captions = split_frames(video)
  summary = condensation(local_captions)
  print(summary)
  text_to_audio(summary)

def text_to_audio(summary):
    tts = gTTS(text=summary, lang='en')
    filename = 'output.mp3'
    tts.save(filename)
    time.sleep(3)
    return filename
    #print('playing sound')
    #playsound(filename)
    #os.close(filename)

interface_description="""<p>Generative AI for Intelligent Video Summarization</p>
                          <p>Pass any video/webcam feed as input along with the summary format, the model will generate a summary of the video based on the format.</p> 
                          <p>Can perform Video understanding and summarizationand can handle short videos upto 30 seconds.</p>
                            """


def video_identity(video, prompt_type):
    start_time = time.time()
    local_captions, extracted_frames_count, extracted_frames = split_frames(video, prompt_type)

    interval_frame_sampling = {
        "Extracted frames" : extracted_frames_count,
    }

    summary = condensation(local_captions, prompt_type)

    generated_captions = {}
    for i in range(extracted_frames_count):
        generated_captions[str(i)] =  local_captions[i]
    
    audio = text_to_audio(summary)

    runtime = int(time.time() - start_time)
    return (runtime, interval_frame_sampling, extracted_frames, generated_captions, summary, audio)

demo = gr.Interface(video_identity, 
                    inputs=[
                        gr.Video(width=400, height=400, container=True),
                        gr.Radio(["general summary", "poetic summary", "sports commentary"], label="Prompt Type"),
                        ], 
                    outputs=[
                        gr.Number(label="Runtime (seconds)"),
                        gr.JSON(label="Extracted frames using IFS"), 
                        gr.Gallery(label="Extracted_frames"),
                        gr.JSON(label="Generated captions for extracted frames"), 
                        gr.Textbox(label = "Condensed final summary"),
                        gr.Audio(label="Summarized content as audio")
                    ],
                    examples=[["footage1.mp4"],["footage2.mp4"], ["footage3.mp4"], ["footage4.mp4"],["footage5.mp4"],["footage6.mp4"],["footage8.mp4"],["footage9.mp4"], ["footage10.mp4"], ["footage13.mp4"], ["footage12.mp4"], ["footage14.mp4"]],
                    title="GenAI-IVS",
                    description=interface_description,
)
demo.launch(share=True)