// firebase.js

import { initializeApp } from "firebase/app";
import { getFirestore, collection, addDoc } from "firebase/firestore";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyCPhKTVbvY1o4JrLFJzU_u6U1wKl3BSPG4",
  authDomain: "ibvap-ec3d8.firebaseapp.com",
  projectId: "ibvap-ec3d8",
  storageBucket: "ibvap-ec3d8.firebasestorage.app",
  messagingSenderId: "284368947666",
  appId: "1:284368947666:web:a85f70eca15a863c6fa00b",
  measurementId: "G-DLS3ZE0637"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firestore (this is our actual database)
export const db = getFirestore(app);

// Reusable function: saves one event object into the "events" collection
export async function saveEventToFirestore(event) {
  try {
    await addDoc(collection(db, "events"), event);
    console.log("Event saved to Firestore: - firebase.js:27", event.type);
  } catch (err) {
    console.error("FIRESTORE SAVE ERROR: - firebase.js:29", err);
  }
}