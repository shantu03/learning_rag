from langchain_text_splitters import RecursiveCharacterTextSplitter,Language


splitter=RecursiveCharacterTextSplitter(
                                    separators=['\n\n','\n',' ',''],
                                    
                                        chunk_size=200,

                                        chunk_overlap=0)

txt=open('file.txt','r').read()

chunks=splitter.split_text(txt)

print(len(chunks))

for i in chunks:
    print(i)