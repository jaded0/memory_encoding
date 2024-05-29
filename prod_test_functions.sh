#!/bin/bash

echo "Testing add-run function:"
curl -X POST "$PROD_SUPABASE_URL/functions/v1/add-run" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $PROD_SUPABASE_ANON_KEY" \
  -d '{"file": "#!/bin/bash\n\necho \"Hello, World!\""}'

echo "\nTesting get-latest-run function:"
curl -X GET "$PROD_SUPABASE_URL/functions/v1/get-latest-run" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $PROD_SUPABASE_ANON_KEY"

  echo "\nTesting get-latest-run function 2nd time:"
curl -X GET "$PROD_SUPABASE_URL/functions/v1/get-latest-run" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $PROD_SUPABASE_ANON_KEY"