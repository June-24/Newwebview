import express from 'express';
import fetch from 'node-fetch';
import { JSDOM } from 'jsdom';

const app = express();
const port = 5001;

app.use(express.json());

app.post('/content', async (req, res) => {
    try {
        const url = req.body.url; // Access the URL sent from the client
        console.log('URL:', url);

        const response = await fetch(url);
        const html = await response.text();
        
        const dom = new JSDOM(html);
        const document = dom.window.document;

        // Extract content as needed
        const divContent = document.querySelector('.grid-col-12.desktop\\:grid-col-6.desktop\\:padding-right-5.margin-bottom-6.desktop\\:margin-bottom-0')?.outerHTML;

        if (divContent) {
            // console.log(divContent);
            res.send(divContent);
        } else {
            console.log('Div with the specified class name not found');
            res.status(404).send('Div with the specified class name not found');
        }
    } catch (error) {
        console.error('Error fetching the HTML:', error);
        res.status(500).send('Error fetching the HTML content');
    }
});


app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
