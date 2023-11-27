# Application Streamlit

Ce projet d'initiative personnelle vise à créer une application avec une interface graphique, résolvant des cas d'usages type. L'application utilise des LLMs d'Hugging Face, le framework LangChain pour créer l'application (back) et Streamlit pour l'interface utilisateur.



Le Dossier contient :

- un fichier ChatGPT.py (le main)

- un dossier /page/ contenant les 2 autres pages : QA.py et Summerizer.py

- un dossier /documents/ contenant un fichier d'illustration : constitution.py



## ChatGPT

La première application est un chatbot. Son principe est simple : il s'agit d'une conversation entre l'IA et un humain. En temps normal, lorsque qu'on infère un LLM, chaque appel est indépendant. Ainsi, il faut ajouter de la _mémoire_ au modèle. L'idée est très simple : on ajoute dans le prompt l'historique de la conversation.



Il y a 3 fonctions :

- `load_LLM` qui va télécharger _en 8bit_ le LLM demandé sur Hugging Face en local. On créer une `pipeline` regroupant certains paramètres importants : _max_length, temperature, top p_. On utilise `HuggingFacePipeline` de LangChain.

- `chaine` qui va créer la chaine LLM. On y décrit le `prompt` utilisé. On instancie la classe `ConversationBufferMemory` pour créer l'objet `memory`. Il y a d'autres classes qui permettent de créer un ConversationBuffer* (qui permet de changer la manière dont on choisit l'historique de conversation : les _k_ derniers échanges, derniers tokens...). Enfin, on créer la chaine avec la classe `LLMChain`.

- `main` qui contient le cœur de l'application : l'affichage Streamlit. Avec `st.sidebar.success`, on peut afficher les fichiers .py contenus dans le dossier /pages/. Il faut créer des variables de sessions `llm_chain` et `messages` : à chaque action, la page Streamlit se recharge (on veut éviter de recharger ces variables à chaque fois). Avec la méthode `predict`, sur `llm_chain`, on peut inférer le modèle. On veillera bien à ajouter chaque message dans la variable `messages` pour les afficher.



## QA

Cette deuxième application permet d'effectuer du Question/Answering sur des documents. Pour ce faire, on utilise la recherche sémantique entre la question et notre base de données, puis avec la méthode _stuffing_, on insère dans le prompt les morceaux de textes qui semblent les plus pertinents. _NB : il existe un notebook dans `/Stage-Marijan/Notebooks/QA PDF avec LangChain`_. Voici les étapes clés :

1. Découper le(s) document(s) en morceaux

2. Import le modèle d'embeddings, calculer les embeddings des morceaux de document pour les stocker dans une base de données.

3. Créer une chaîne avec la classe `ConversationalRetrievalChain`





Il y a 5 fonctions :

- `load_LLM` qui va télécharger _en 8bit_ le LLM demandé sur Hugging Face en local. On créer une `pipeline` regroupant certains paramètres importants : _max_length, temperature, top p_. On utilise `HuggingFacePipeline` de LangChain.

- `load_document`, objet de la classe `RecursiveCharacterTextSplitter`, permet de découper un long texte initial, en plus petits morceaux (plusieurs méthodes possibles). Elle renvoie une liste de morceaux de textes. La taille de ces morceaux doit être bien pensée : on calculera les embeddings de ces morceaux.

- `qa` qui va créer la chaîne grâce à la classe `ConversationalRetrievalChain`. L'objet `memory` n'est malheureusement pas utile ni utilisable : on ne peut poser qu'une seule question à la fois. `retriever` décrit la manière dont on souhaite récupérer la donnée : recherche par similarité sémantique (calcul du cosinus), en sélectionnant les 3 meilleurs résultats. Le paramètre `return_source_documents` permet de renvoyer la source du document.

- `process_llm_response` permet de mettre en forme l'output de la chaîne de QA (qui est un dictionnaire). Elle renvoie une liste contenant les informations de la source (numéro de page et localisation du ficher) ainsi que la réponse de la chaîne

- `main` qui contient le cœur de l'application : l'affichage Streamlit. Il faut créer des variables de sessions `local_llm`, `messages_qa` (historique), `qa` (chaine de QA), `embedding` (modèle d'embedding). Pour éviter les problèmes de variables, j'ai mis le cœur des étapes dans le `main` (pratique qu'il ne faut pas faire). Dans un premier temps, on charge tous les .pdf du dossier /document/. _NB : Il était plus délicat de faire cette application en ajoutant un espace de dépôt de fichier avec streamlit : en effet, nous n'avons accès qu'au texte (raw) il manque donc des informations comme la page, etc... Ce qui ne permet pas de remonter à la source._ Ensuite, on créer notre liste de morceaux de texte avec `load_document`. On télécharge un modèle d'embedding d'HuggingFace. On créé ensuite `vectordb` avec la librairie `Chroma`, qui contiendra les embeddings des morceaux de textes. Enfin, pour une question donnée, on peut inférer le modèle avec la chaine `qa`.





## Summerizer

Cette dernière application permet de synthétiser un document texte. En utilisant une technique au choix :

- _map reduce_ : On résume dans une première phase chaque morceau de texte (appels au LLM parallélisables). Puis on résume la concaténation de ces résumés.

- _refine_ : On résume le premier morceau de texte. Puis on résume la concaténation de ce résumé avec le deuxième morceau. Ainsi de suite jusqu'à obtenir un résumé final (appels au LLM en série).







Il y a 4 fonctions :

- `load_LLM` qui va télécharger _en 8bit_ le LLM demandé sur Hugging Face en local. On créer une `pipeline` regroupant certains paramètres importants : _max_length, temperature, top p_. On utilise `HuggingFacePipeline` de LangChain.

- `load_document`, objet de la classe `RecursiveCharacterTextSplitter`, permet de découper un long texte intial, en plus petits morceaux (plusieurs méthodes possibles). Elle renvoie une liste de morceaux de textes (qui sont des `Document`).

- `load_chain` qui va créer la chaîne grâce à la classe `ConversationalRetrievalChain`. On y retrouve les 2 prompts utilisés (pour le map reduce, ne pas les inclure pour le refine). Avec `verbose` on peut observer étape par étape le processus de map reduce.

- `get_pdf_text` permet de récupérer le contenu texte d'un fichier (.pdf ou .txt) déposé par l'utilisateur.

- `main` qui contient le cœur de l'application : l'affichage Streamlit. Il faut créer des variables de sessions `local_llm`, `chain` (chaine de summarizer). Avec `file_uploader` on récupère le fichier de l'utilisateur. Ensuite, on extrait le contenu texte, et on le découpe en morceaux. On créé la chaîne avec `load_chain`. Enfin, on affiche (dans la pseudo-conversation avec le LLM) d'abord le texte (attention, il peut être très long !) puis le résultat de la chaîne avec la méthode `run`.