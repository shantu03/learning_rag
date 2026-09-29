from langchain_community.document_loaders import YoutubeLoader


loader=YoutubeLoader.from_youtube_url(youtube_url='https://www.youtube.com/watch?v=rDteJAuKiBI')

video_transit=loader.load()

print(video_transit)
