mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"your-email@example.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
[client]\n\
showErrorDetails = true\n\
maxUploadSize = 1024\n\
" > ~/.streamlit/config.toml
