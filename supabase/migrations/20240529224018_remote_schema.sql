alter table "public"."runs" drop column "started";

alter table "public"."runs" add column "started_at" timestamp with time zone;


