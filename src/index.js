import { ChatGroq } from "@langchain/groq";
import { OllamaEmbeddings } from "@langchain/ollama";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { TavilySearchAPIRetriever } from "@langchain/community/retrievers/tavily_search_api";
import express from 'express'
import dotenv from 'dotenv'

dotenv.config({path:'src/.env'});

const app = express()

app.use(express.json())
app.use(express.urlencoded({ extended:true }))

const GROQ_API_KEY = process.env.GROQ_API_KEY;
const TAVILY_API = process.env.TAVILY_SEARCH_API;


const model = new ChatGroq({
    model: "llama-3.3-70b-versatile", // Recommended model 
    temperature: 0,
    apiKey: GROQ_API_KEY,
});

const promptRealOrRag = `
Analyze the following question. If it contains references to recent events, real-time data, trending topics, or currently unfolding situations (e.g., today's news, current year, live updates, recent technology releases, ongoing sports events), then respond with "real". Otherwise, if it's a general, timeless, or static question, respond with "rag".
question:
{question}
`
const standardQuestionPromptText = `Given a question,convert it to a standard question format Question \n  {question}`;

const answerPromptText = `You are a helpful and enthusiastic support bot. Your job is to answer the given question using the context provided. The context includes both document content and associated source links (called 'metasource').

Always answer the question based only on the content. If you do not know the answer, respond with:
"I'm sorry, I don't know the answer to that question. Please check whether you provided the document related to that question."

When responding:
- Provide a clear, concise answer based only on the content.
- On a new line, list any related links from the 'metasource' that support your answer.
- If a source appears multiple times, only include it once in the list (no duplicates).

Format:
<your answer> 

Sources:
<unique, relevant links from metasource (one per line)>

Context:
{context}

Question:
{question}

Answer:`;


const realOrRagPrompt = ChatPromptTemplate.fromTemplate(promptRealOrRag);
  
const realOrRagChain = realOrRagPrompt.pipe(model).pipe(new StringOutputParser());
  
const standardQuestionPrompt = ChatPromptTemplate.fromTemplate(standardQuestionPromptText);
  
const standardQuestionChain =  standardQuestionPrompt.pipe(model).pipe(new StringOutputParser())
  
const answerPrompt = ChatPromptTemplate.fromTemplate(answerPromptText).pipe(model)
  
const chatWithModel = async (query)=>{
  // Acts like an agent route
  // when we ask a question based on that it will route to RAG
  // OR it will route to the real time data search from web
  const result = await realOrRagChain.invoke({question: query});
  console.log(result,"res", result.includes('rag'))
  const docPath = "./nodejs_tutorial.pdf";
    
   const loader = new PDFLoader(docPath);
    
    const docs = await loader.load();
    
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 100,
    });
    
    const texts = await textSplitter.splitDocuments(docs);
    
    const embeddings = new OllamaEmbeddings({
      model: "mxbai-embed-large",
      baseUrl: "http://127.0.0.1:11434", 
    });
    
    const vectorStore = await FaissStore.fromDocuments(texts, embeddings);
    
    const retriever_vect = vectorStore.asRetriever()
    
    const retriever_Tavily = new TavilySearchAPIRetriever({
      k: 3,
      apiKey:TAVILY_API
    });
    
    const standardRetriver =  RunnableSequence.from([
      standardQuestionChain,
      prev => prev,
      result.includes('rag') === true || result === 'rag' ? retriever_vect : retriever_Tavily,
      (input)=> {
        // console.log(input)
        const content = {
           pageContent: input.map(docs => docs.pageContent).join('\n'),
           metaSource: input.map(docs => docs.metadata.source !== undefined && docs.metadata.source)
        }
        console.log(content)
        return content
      },
    ])
    
    const standardChain = RunnableSequence.from([
      {
        question:standardQuestionChain,
        context:standardRetriver
      },
      answerPrompt,
      new StringOutputParser(),
    ])
    
    const standardans = await standardChain.invoke({ question: query });
    return standardans;
}

app.get('/', (req,res)=>{
   res.send("Hello")
})

app.post('/chat', async (req,res)=>{
     const { query } = req.body;
     if(query){
       const ans = await chatWithModel(query);
       return res.status(200).json({answer: ans})
     }

     return res.status(500).json({ message:"no query" })
})

app.listen(3000, ()=>{
   console.log('server is connected.')
})