---


---

<h1 id="ainanda---ai-based-sentiment-analysis-to-classify-text-based-on-its-positivity-level">AInanda - AI based sentiment analysis to classify text based on its positivity level</h1>
<p><img src="https://s18.directupload.net/images/210418/b3ns2zgo.png" alt="enter image description here"></p>
<p><strong>Problem Statement:</strong><br>
In times of COVID-19, increasing focus on digital communication and negativity is surrounding us. Negative news can be a rabbit hole, people get lost in bad thoughts and loose emotional intelligence in virtual communication. With AInanda we want to reverse and counteract this trend. Before sending text, one can check the positivity of written communication with the AI based sentiment analysis.</p>
<blockquote>
<p><strong>Team Members;</strong> Kristina Klein, Melih Yozgyur, Inês Santos, Mert Özcan</p>
</blockquote>
<h2 id="table-of-contents">Table of Contents</h2>
<ol>
<li><a href="#introduction">Introduction</a></li>
<li><a href="#data-preprocessing">Data preprocessing</a></li>
<li><a href="#vectorization">Vectorization</a></li>
<li><a href="#prepare-&amp;-training-the-model">Prepare &amp; Training the model</a></li>
<li><a href="#evaluation">Evaluation</a></li>
</ol>
<h2 id="introduction">Introduction</h2>
<p>AInanda, the Goddess of Happiness is gonna evaluate the sentiment of your text</p>
<ul>
<li>The model was trained with Twitter text data, in total approx. 1.600.000  labeled data from <a href="https://www.kaggle.com/kazanova/sentiment140">https://www.kaggle.com/kazanova/sentiment140</a>]</li>
<li>It runs in Google colab (GPU enabled environment)</li>
<li>Natural language analysis decodes sentiment in three categories ;   <strong>Positive</strong>: 😊, <strong>Neutral</strong>: 😐, <strong>Negative</strong>: 🙁</li>
</ul>
<p>The repository includes:</p>
<ul>
<li>Instructions on how to train from scratch your network</li>
<li>Training and evaluation (accuracy) notebook</li>
<li>Inferencing notebook</li>
</ul>
<p><img src="https://s18.directupload.net/images/210331/ccluj8jc.png" alt="enter image description here"><br>
First we start with loading our dataset. After that we have to clean out dataset in order to use in our model. To do that, we have applied several aprroaches, like stemming or stopwords to our dataset. After the data is cleaned it needs to be vectorised with the aim of finding  words with similar meanings in our dataset. Subsequently we train our model and make our predictions.</p>
<h3 id="loading-dataset">Loading Dataset</h3>
<p>Downloading Kaggle datasets via Kaggle API</p>
<p><strong>1 – Get the API key from your account</strong></p>
<ul>
<li><a href="http://www.kaggle.com">www.kaggle.com</a> ⇨ login ⇨ My Account ⇨ Create New API Token</li>
<li>The “kaggle.json” file will be auto downloaded.</li>
</ul>
<p><strong>2 — Upload the kaggle.json file</strong><br>
<strong>3 — Download the required dataset and unzip</strong></p>
<ul>
<li><code>!kaggle datasets download -d kazanova/sentiment140</code></li>
<li><code>!unzip sentiment140.zip</code></li>
</ul>
<h2 id="data-preprocessing">Data Preprocessing</h2>
<h3 id="data-cleaning">Data cleaning</h3>
<p>In order to get accurate results from our model, first we need to clean and organise out data. Special characters like ‘@‘ number, punctuations, hashtag, URL, HTML or CSS elements from data were eliminated. On the other hand the text data is converted to lowercase to help with the process of preprocessing and later in model training.<br>
<img src="https://analyticsindiamag.com/wp-content/uploads/2020/09/Data-Cleaner.png" alt="enter image description here"></p>
<h3 id="stopwords">Stopwords</h3>
<p><strong>Stopwords</strong> are the <strong>words</strong> in any language that does not add much meaning to a sentence.<br>
<img src="https://media.geeksforgeeks.org/wp-content/cdn-uploads/Stop-word-removal-using-NLTK.png" alt="enter image description here"><br>
That is why they are filtered out before or after the natural language data (text) are processed. While “stop words” typically refer to the most common words in a language, all-natural language processing tools don’t use a single common list of stop words. Removing stop words helps to decrease the size of the data set and the time to train the model. While excluding stopwords, one should be careful.</p>
<blockquote>
<p>“ The weather is not good.”</p>
</blockquote>
<p>If we remove (not ) in preprocessing step the sentence (the weather is good) means that it is positive which is wrongly interpreted.</p>
<blockquote></blockquote>
<h3 id="tokenization">Tokenization</h3>
<p>Tokenization is the process of breaking down a piece of text into small sections. Tokens can be a word, part of a word or just characters like punctuation. Tokenization defines what our NLP(Natural Language Processing) models can express.<br>
Languages have distinct strings that have meaning. They called words in our world. But a token is a string with a known meaning in NLP<br>
<img src="https://s20.directupload.net/images/210331/5zi8emri.png" alt="Tokenization"></p>
<p>Algorithms like <a href="https://en.wikipedia.org/wiki/Word2vec">Word2Vec</a> or <a href="https://nlp.stanford.edu/projects/glove/">GloVe</a> assign a vector to a token. You can do “King-man +woman” and get the vector for queen.<br>
<img src="https://www.machinelearningplus.com/wp-content/uploads/2021/02/vector.png" alt="enter image description here"></p>
<h3 id="stemming">Stemming</h3>
<p>Stemming (stem form reduction) is the term used in information retrieval as well as in linguistic computer science to describe a procedure by which different morphological variants of a word are reduced to their common root.</p>
<p>The stem of the  verb  <strong>wait</strong>  is  <strong>wait</strong>: it is the part that is common to all its inflected variants.</p>
<ol>
<li><strong>wait</strong>  (infinitive)</li>
<li><strong>wait</strong>  (imperative)</li>
<li><strong>wait</strong>s (present, 3rd person, singular)</li>
<li><strong>wait</strong>  (present, other persons and/or plural)</li>
<li><strong>wait</strong>ed (simple past)</li>
<li><strong>wait</strong>ed (past participle)</li>
<li><strong>wait</strong>ing (progressive)</li>
</ol>
<p>There are two types of stemmers, on the one hand the algorithmic stemmers and on the other hand the stemmers based on a dictionary.</p>
<h4 id="over-and-understemming">Over and Understemming</h4>
<p>Under-stemming is an error which in itself cannot harm any Information Retrieval system, the result is always better than without stemming. Over-stemming is a more serious problem, but we don’t care about that either, as long as two words of different meaning don’t coincide, and this case is not too frequent. Only under stemming is critical, but its occurrence can be reduced by a dictionary. These considerations may help to explain why algorithmic stemming methods produce such good results in spite of everything.</p>
<h2 id="vectorization">Vectorization</h2>
<p>Vectorization is a technique by  which you can make your code execute fast. It is a important way to optimize algorithms when you are implementing it from scratch. By using a vectorized implementation in an optimization algorithm we can make the process of computation much faster compared to unvectorized Implementation.</p>
<p>Word2Vec is one of the most popular technique to learn word embeddings using shallow neural network. It was developed by Tomas Mikolov in 2013 at Google</p>
<p>Common Bag Of Words (CBOW) method in Word2Vec. In this model text is served as a bag with many words and it describes the occurrence of words within a document. The position of the words in the sentence doesn’t matter.</p>
<h2 id="prepare--training-the-model">Prepare &amp; Training the model</h2>
<ul>
<li>We will continue to use trained word embeddings to represent words with word2vec and then we will feed them into an LSTM</li>
<li>LSTM (long short term memory) save the words and predict the next words based on the previous words. LSTM is a sequance predictor of next coming words.</li>
</ul>
<p><img src="https://s16.directupload.net/images/210331/aeyjswlq.png" alt="enter image description here"></p>
<h2 id="evaluation">Evaluation</h2>
<ul>
<li><a href="https://github.com/magicmercedes/ainanda-text-sentiment-analysis/blob/main/ainanda_inference.ipynb">ainanda_inference.ipynb</a> to visualize the results through colab Notebook. Some results are shown as follows:</li>
</ul>
<p><strong>Positive</strong></p>
<pre><code>predict_text("TechLabs is one the best place to enter the coding world")
{'Emoji': '😊', 'Label': 'Positive', 'Score': 0.9134523868560791}
</code></pre>
<p><strong>Negative</strong></p>
<pre><code>predict_text("My package hasn’t arrived yet.")
{'Emoji': '🙁', 'Label': 'Negative', 'Score': 0.24372592568397522}
</code></pre>
<p><strong>Neutral</strong></p>
<pre><code>predict_text("they should be here")
{'Emoji': '😐', 'Label': 'Neutral', 'Score': 0.5754445791244507}
</code></pre>

