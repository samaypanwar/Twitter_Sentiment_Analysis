def emoji_sentiment(texts):
    import numpy as np
    from emosent import get_emoji_sentiment_rank

    # this function is there to analyse and return the emoji sentiment per text in a list of emoji-filled text docs
    emoji=[] #emoji is a list of all the emoji sentiments found in a list of documents
    avg_emoji_sentiment=[] #avg_ emoji sent is a list of the average sentiment defined by emojis in a list of docs

    for text in texts:

        doc=text.split()
        avg_emoji_sentiment.append(0.0) # we use 0.0 as the neutral sentiment here and if the doc does not contain any emoji or 
                                        #if the emoji is not recognised by the function then 0.0 gets appended to the 'emoji' list
        emoji.append([])

        for word in doc:
            try:
                emoji[-1].append(get_emoji_sentiment_rank(word))
            except:
                pass

        if emoji[-1]:
            avg_emoji_sentiment[-1] = round(np.mean(emoji[-1]),3)
        #returns a list of the avg emoji sentiment per document in a list of docs
    return avg_emoji_sentiment