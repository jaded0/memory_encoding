#!/bin/bash

echo "Testing add-run function:"
curl -X POST 'http://localhost:54325/functions/v1/add-run' \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $SUPABASE_SERVICE_ROLE_KEY" \
  -d '{"file": "#!/bin/bash\n\necho \"Hello, World!\""}'

echo "\nTesting get-latest-run function:"
curl -X GET 'http://localhost:54325/functions/v1/get-latest-run' \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $SUPABASE_SERVICE_ROLE_KEY"