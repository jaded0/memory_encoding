import { serve } from 'https://deno.land/std/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

serve(async (req) => {
  const supabaseClient = createClient(
    Deno.env.get('SUPABASE_URL') ?? '',
    Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
  )

  const { data: latestRun, error } = await supabaseClient
    .from('runs')
    .select('id, file')
    .is('started_at', null) // Use .is() instead of .eq() for checking null values
    .order('created_at', { ascending: false })
    .limit(1)
    .single()

  if (error) {
    return new Response(JSON.stringify({ error: error.message }), {
      headers: { 'Content-Type': 'application/json' },
      status: 400,
    })
  }

  if (!latestRun) {
    return new Response(JSON.stringify({ message: 'No pending runs found' }), {
      headers: { 'Content-Type': 'application/json' },
      status: 404,
    })
  }

  await supabaseClient
    .from('runs')
    .update({ started_at: new Date().toISOString() })
    .eq('id', latestRun.id)

  return new Response(JSON.stringify(latestRun), {
    headers: { 'Content-Type': 'application/json' },
    status: 200,
  })
})



// Follow this setup guide to integrate the Deno language server with your editor:
// https://deno.land/manual/getting_started/setup_your_environment
// This enables autocomplete, go to definition, etc.

// console.log("Hello from Functions!")

// Deno.serve(async (req) => {
//   const { name } = await req.json()
//   const data = {
//     message: `Hello ${name}!`,
//   }

//   return new Response(
//     JSON.stringify(data),
//     { headers: { "Content-Type": "application/json" } },
//   )
// })

// To invoke:
// curl -i --location --request POST 'http://localhost:54321/functions/v1/' \
//   --header 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0' \
//   --header 'Content-Type: application/json' \
//   --data '{"name":"Functions"}'
